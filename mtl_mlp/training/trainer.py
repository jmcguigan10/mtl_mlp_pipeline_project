from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from ..utils.common import count_parameters, ensure_dir, get_device, move_batch_to_device, prune_checkpoints, save_json, set_seed
from .balancers import GradNormLossBalancer, StaticLossBalancer, build_loss_balancer
from .losses import TaskLossBundle, build_loss_bundle
from .optim import SchedulerBundle, build_optimizers, build_scheduler
from .pcgrad import PCGrad


@dataclass
class EpochAccumulator:
    bc_threshold: float = 0.5

    def __post_init__(self) -> None:
        self.loss_sums: dict[str, float] = {}
        self.num_samples = 0
        self.bc_correct = 0
        self.bc_tp = 0
        self.bc_fp = 0
        self.bc_fn = 0
        self.bc_tn = 0
        self.reg_abs = 0.0
        self.reg_sq = 0.0
        self.reg_count = 0
        self.vec_abs = 0.0
        self.vec_sq = 0.0
        self.vec_l2 = 0.0
        self.vec_count = 0
        self.vec_items = 0

    def update_losses(self, losses: dict[str, float], batch_size: int) -> None:
        self.num_samples += batch_size
        for key, value in losses.items():
            self.loss_sums[key] = self.loss_sums.get(key, 0.0) + float(value) * batch_size

    def update_outputs(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        bc_logits = outputs['bc'].detach()
        bc_target = batch['bc_target'].detach()
        if bc_logits.shape[-1] == 1:
            probs = torch.sigmoid(bc_logits.reshape_as(bc_target))
            preds = (probs >= self.bc_threshold).long()
            targets = bc_target.long()
            self.bc_correct += int((preds == targets).sum().item())
            self.bc_tp += int(((preds == 1) & (targets == 1)).sum().item())
            self.bc_fp += int(((preds == 1) & (targets == 0)).sum().item())
            self.bc_fn += int(((preds == 0) & (targets == 1)).sum().item())
            self.bc_tn += int(((preds == 0) & (targets == 0)).sum().item())
        else:
            preds = torch.argmax(bc_logits, dim=-1)
            targets = bc_target.view(-1).long()
            self.bc_correct += int((preds == targets).sum().item())

        reg_pred = outputs['regression'].detach()
        reg_target = batch['reg_target'].detach()
        reg_diff = reg_pred - reg_target
        self.reg_abs += float(reg_diff.abs().sum().item())
        self.reg_sq += float((reg_diff ** 2).sum().item())
        self.reg_count += int(reg_diff.numel())

        vec_pred = outputs['vector_regression'].detach()
        vec_target = batch['vector_target'].detach()
        vec_diff = vec_pred - vec_target
        self.vec_abs += float(vec_diff.abs().sum().item())
        self.vec_sq += float((vec_diff ** 2).sum().item())
        self.vec_l2 += float(torch.norm(vec_diff, dim=-1).sum().item())
        self.vec_count += int(vec_diff.shape[0])
        self.vec_items += int(vec_diff.numel())

    def summarize(self, prefix: str) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for key, value in self.loss_sums.items():
            metrics[f'{prefix}/{key}'] = value / max(self.num_samples, 1)
        total_bc = self.bc_tp + self.bc_fp + self.bc_fn + self.bc_tn
        if total_bc > 0:
            precision = self.bc_tp / max(self.bc_tp + self.bc_fp, 1)
            recall = self.bc_tp / max(self.bc_tp + self.bc_fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            metrics[f'{prefix}/bc_accuracy'] = self.bc_correct / max(total_bc, 1)
            metrics[f'{prefix}/bc_precision'] = precision
            metrics[f'{prefix}/bc_recall'] = recall
            metrics[f'{prefix}/bc_f1'] = f1
        metrics[f'{prefix}/reg_mae'] = self.reg_abs / max(self.reg_count, 1)
        metrics[f'{prefix}/reg_mse'] = self.reg_sq / max(self.reg_count, 1)
        metrics[f'{prefix}/reg_rmse'] = math.sqrt(metrics[f'{prefix}/reg_mse'])
        metrics[f'{prefix}/vector_mae'] = self.vec_abs / max(self.vec_items, 1)
        metrics[f'{prefix}/vector_mse'] = self.vec_sq / max(self.vec_items, 1)
        metrics[f'{prefix}/vector_rmse'] = math.sqrt(metrics[f'{prefix}/vector_mse'])
        metrics[f'{prefix}/vector_avg_l2'] = self.vec_l2 / max(self.vec_count, 1)
        return metrics


class Trainer:
    def __init__(
        self,
        config: Any,
        model: torch.nn.Module,
        train_loader: Any | None,
        val_loader: Any | None = None,
        test_loader: Any | None = None,
        loss_bundle: TaskLossBundle | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_bundle = loss_bundle or build_loss_bundle(config)
        self.balancer = build_loss_balancer(config, self.loss_bundle.task_specs)
        self.device = get_device(config.training.get('device', 'auto'))
        self.use_pcgrad = str(config.multitask.gradient_surgery.get('name', 'none')).lower() == 'pcgrad'
        self.amp_enabled = bool(config.training.get('mixed_precision', True)) and self.device.type == 'cuda'
        if self.use_pcgrad or isinstance(self.balancer, GradNormLossBalancer):
            self.amp_enabled = False
        self.model.to(self.device)
        self.loss_bundle.to(self.device)
        self.balancer.to(self.device)

        if bool(config.training.get('compile_model', False)) and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)

        self.optimizer, self.balancer_optimizer = build_optimizers(config, self.model, self.balancer)
        steps_per_epoch = len(train_loader) if train_loader is not None else 1
        self.scheduler_bundle: SchedulerBundle = build_scheduler(config, self.optimizer, steps_per_epoch)
        self.pcgrad = PCGrad(reduction=str(config.multitask.gradient_surgery.get('reduction', 'mean'))) if self.use_pcgrad else None
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp_enabled)
        self.grad_clip_norm = config.training.get('grad_clip_norm')
        self.accumulation_steps = int(config.training.get('gradient_accumulation_steps', 1))
        self.bc_threshold = float(config.evaluation.get('bc_threshold', 0.5))
        self.best_monitor_value: float | None = None
        self.best_epoch: int | None = None
        self.history: list[dict[str, float]] = []

        base_output_dir = ensure_dir(config.output.dir)
        experiment_name = str(config.output.get('experiment_name', 'default_run'))
        self.run_dir = ensure_dir(base_output_dir / experiment_name)
        self.checkpoint_dir = ensure_dir(self.run_dir / 'checkpoints')
        self.history_path = self.run_dir / 'history.csv'
        self.metrics_path = self.run_dir / 'final_metrics.json'
        self.config_snapshot_path = self.run_dir / 'config_snapshot.json'
        save_json(config.to_dict(), self.config_snapshot_path)

        self.parameter_counts = count_parameters(self.model)

    @property
    def monitor_name(self) -> str:
        return str(self.config.training.early_stopping.get('monitor', 'val/total_loss'))

    @property
    def monitor_mode(self) -> str:
        return str(self.config.training.early_stopping.get('mode', 'min')).lower()

    def _is_better(self, value: float, best: float | None) -> bool:
        if best is None:
            return True
        if self.monitor_mode == 'max':
            return value > best
        return value < best

    def _write_history_row(self, row: dict[str, float]) -> None:
        self.history.append(row)
        fieldnames = sorted({key for entry in self.history for key in entry.keys()})
        with self.history_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.history:
                writer.writerow(entry)

    def _save_checkpoint(self, epoch: int, is_best: bool, metrics: dict[str, float]) -> None:
        payload = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss_bundle_state_dict': self.loss_bundle.state_dict(),
            'balancer_state_dict': self.balancer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'balancer_optimizer_state_dict': self.balancer_optimizer.state_dict() if self.balancer_optimizer else None,
            'scheduler_state_dict': self.scheduler_bundle.scheduler.state_dict() if self.scheduler_bundle.scheduler else None,
            'metrics': metrics,
            'config': self.config.to_dict(),
        }
        torch.save(payload, self.checkpoint_dir / 'last.pt')
        if bool(self.config.logging.get('save_every_epoch', True)):
            torch.save(payload, self.checkpoint_dir / f'epoch_{epoch:03d}.pt')
            prune_checkpoints(self.checkpoint_dir, int(self.config.logging.get('keep_last_n_checkpoints', 3)))
        if is_best:
            torch.save(payload, self.checkpoint_dir / 'best.pt')

    def _step_scheduler(self, metric_value: float | None = None) -> None:
        scheduler = self.scheduler_bundle.scheduler
        if scheduler is None or self.scheduler_bundle.step_on == 'batch':
            return
        if self.scheduler_bundle.step_on == 'metric':
            if metric_value is None:
                raise ValueError('Metric scheduler requires a monitor value.')
            scheduler.step(metric_value)
        else:
            scheduler.step()

    def _compute_task_losses(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        outputs = self.model(batch['inputs'])
        task_losses = self.loss_bundle(outputs, batch)
        return outputs, task_losses

    def _backward_standard(self, total_loss: torch.Tensor) -> None:
        if self.amp_enabled:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

    def _optimizer_step_standard(self) -> None:
        if self.grad_clip_norm is not None:
            if self.amp_enabled:
                self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), float(self.grad_clip_norm))
        if self.amp_enabled:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def _train_batch(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        self.optimizer.zero_grad(set_to_none=True)
        if self.balancer_optimizer is not None:
            self.balancer_optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda', enabled=self.amp_enabled):
            outputs, task_losses = self._compute_task_losses(batch)

        detached_log = {name: float(loss.detach().item()) for name, loss in task_losses.items()}

        if self.use_pcgrad:
            assert isinstance(self.balancer, StaticLossBalancer)
            weighted_losses = self.balancer.weighted_losses(task_losses)
            total_loss = sum(weighted_losses.values())
            self.pcgrad.pc_backward([weighted_losses[name] for name in self.balancer.task_names], self.model.parameters())
            if self.grad_clip_norm is not None:
                clip_grad_norm_(self.model.parameters(), float(self.grad_clip_norm))
            self.optimizer.step()
            weight_info = self.balancer.current_weight_dict()
            detached_log['total_loss'] = float(total_loss.detach().item())
            detached_log.update({f'weight/{k}': v for k, v in weight_info.items()})
            return outputs, detached_log

        if isinstance(self.balancer, GradNormLossBalancer):
            total_loss, weight_info = self.balancer.aggregate(task_losses)
            total_loss.backward(retain_graph=True)
            weight_grads, gradnorm_stats = self.balancer.compute_weight_gradients(task_losses, self.model.get_shared_parameters())
            if self.grad_clip_norm is not None:
                clip_grad_norm_(self.model.parameters(), float(self.grad_clip_norm))
            self.optimizer.step()
            assert self.balancer_optimizer is not None
            self.balancer_optimizer.zero_grad(set_to_none=True)
            self.balancer.weights.grad = weight_grads
            self.balancer_optimizer.step()
            self.balancer.renormalize_()
            detached_log['total_loss'] = float(total_loss.detach().item())
            detached_log.update({f'weight/{k}': v for k, v in self.balancer.current_weight_dict().items()})
            detached_log.update(gradnorm_stats)
            return outputs, detached_log

        total_loss, weight_info = self.balancer.aggregate(task_losses)
        self._backward_standard(total_loss)
        self._optimizer_step_standard()
        detached_log['total_loss'] = float(total_loss.detach().item())
        detached_log.update({f'weight/{k}': v for k, v in weight_info.items()})
        return outputs, detached_log

    def _evaluate_batch(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        with torch.no_grad():
            outputs, task_losses = self._compute_task_losses(batch)
            total_loss, weight_info = self.balancer.aggregate(task_losses)
        detached_log = {name: float(loss.detach().item()) for name, loss in task_losses.items()}
        detached_log['total_loss'] = float(total_loss.detach().item())
        detached_log.update({f'weight/{k}': v for k, v in weight_info.items()})
        return outputs, detached_log

    def _run_loader(self, loader: Any, training: bool, epoch: int) -> dict[str, float]:
        if loader is None:
            return {}
        if training:
            self.model.train(True)
        else:
            self.model.train(False)

        prefix = 'train' if training else 'val'
        accumulator = EpochAccumulator(bc_threshold=self.bc_threshold)
        progress = tqdm(loader, desc=f'{prefix} epoch {epoch}', leave=False)

        for step, raw_batch in enumerate(progress, start=1):
            batch = move_batch_to_device(raw_batch, self.device)
            if training:
                outputs, detached_log = self._train_batch(batch)
                if self.scheduler_bundle.scheduler is not None and self.scheduler_bundle.step_on == 'batch':
                    self.scheduler_bundle.scheduler.step()
            else:
                outputs, detached_log = self._evaluate_batch(batch)

            batch_size = int(batch['inputs'].shape[0])
            accumulator.update_losses(detached_log, batch_size)
            accumulator.update_outputs(outputs, batch)

            if step % int(self.config.logging.get('train_log_interval', 20)) == 0 or step == len(loader):
                progress.set_postfix({'total_loss': f"{detached_log['total_loss']:.4f}"})

        return accumulator.summarize(prefix)

    def fit(self) -> dict[str, float]:
        seed = int(self.config.get('seed', 42))
        deterministic = bool(self.config.training.get('deterministic', False))
        set_seed(seed, deterministic=deterministic)

        patience = int(self.config.training.early_stopping.get('patience', 10))
        early_stopping_enabled = bool(self.config.training.early_stopping.get('enabled', True))
        bad_epochs = 0
        final_metrics: dict[str, float] = {}

        for epoch in range(1, int(self.config.training.epochs) + 1):
            train_metrics = self._run_loader(self.train_loader, training=True, epoch=epoch) if self.train_loader else {}
            val_metrics = self._run_loader(self.val_loader, training=False, epoch=epoch) if self.val_loader else {}
            merged_metrics = {'epoch': float(epoch), **train_metrics, **val_metrics}

            monitor_value = merged_metrics.get(self.monitor_name)
            if monitor_value is None and train_metrics:
                monitor_value = merged_metrics.get('train/total_loss')

            if monitor_value is not None:
                is_best = self._is_better(monitor_value, self.best_monitor_value)
                if is_best:
                    self.best_monitor_value = monitor_value
                    self.best_epoch = epoch
                    bad_epochs = 0
                else:
                    bad_epochs += 1
            else:
                is_best = False

            if self.optimizer.param_groups:
                merged_metrics['lr'] = float(self.optimizer.param_groups[0]['lr'])
            self._write_history_row(merged_metrics)
            self._save_checkpoint(epoch=epoch, is_best=is_best, metrics=merged_metrics)
            self._step_scheduler(metric_value=monitor_value)

            if early_stopping_enabled and bad_epochs >= patience:
                final_metrics = merged_metrics
                break
            final_metrics = merged_metrics

        if self.test_loader is not None:
            test_metrics = self.evaluate(self.test_loader, split_name='test')
            final_metrics.update(test_metrics)

        final_metrics['best_epoch'] = float(self.best_epoch or 0)
        final_metrics['parameters/total'] = float(self.parameter_counts['total'])
        final_metrics['parameters/trainable'] = float(self.parameter_counts['trainable'])
        save_json(final_metrics, self.metrics_path)
        return final_metrics

    def evaluate(self, loader: Any | None = None, split_name: str = 'eval') -> dict[str, float]:
        loader = loader or self.val_loader or self.test_loader
        if loader is None:
            return {}
        self.model.train(False)
        accumulator = EpochAccumulator(bc_threshold=self.bc_threshold)
        progress = tqdm(loader, desc=split_name, leave=False)
        for raw_batch in progress:
            batch = move_batch_to_device(raw_batch, self.device)
            outputs, detached_log = self._evaluate_batch(batch)
            batch_size = int(batch['inputs'].shape[0])
            accumulator.update_losses(detached_log, batch_size)
            accumulator.update_outputs(outputs, batch)
        return accumulator.summarize(split_name)
