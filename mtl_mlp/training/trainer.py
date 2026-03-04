from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from ..preprocessing import Box3DHeuristic
from ..utils.common import (
    count_parameters,
    ensure_dir,
    get_device,
    load_torch_checkpoint,
    move_batch_to_device,
    prune_checkpoints,
    save_json,
    set_seed,
)
from .balancers import GradNormLossBalancer, StaticLossBalancer, build_loss_balancer
from .epoch_metrics import EpochAccumulator
from .losses import TaskLossBundle, build_loss_bundle
from .optim import SchedulerBundle, build_optimizers, build_scheduler
from .pcgrad import PCGrad


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
        self._use_new_amp = bool(
            hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler') and hasattr(torch.amp, 'autocast')
        )
        if self._use_new_amp:
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp_enabled)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.grad_clip_norm = config.training.get('grad_clip_norm')
        self.accumulation_steps = int(config.training.get('gradient_accumulation_steps', 1))
        self.bc_threshold = float(config.evaluation.get('bc_threshold', 0.5))
        self.control_enabled = bool(config.evaluation.get_path('control.enabled', False))
        self.control_ratio_eps = float(config.evaluation.get_path('control.ratio_eps', 1.0e-8))
        self.control_ratio_floor_quantile = float(config.evaluation.get_path('control.ratio_floor_quantile', 0.10))
        self.control_min_baseline_error = float(config.evaluation.get_path('control.min_baseline_error', 1.0e-4))
        self.control_compute_during_fit = bool(config.evaluation.get_path('control.compute_during_fit', False))
        self.control_input_is_normalized = bool(config.evaluation.get_path('control.input_is_normalized', True))
        self.control_nf = int(config.evaluation.get_path('control.nf', 3))
        self.control_model: Box3DHeuristic | None = None
        if self.control_enabled:
            self.control_model = Box3DHeuristic(self.control_nf).to(self.device)
            self.control_model.eval()
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

    def _restore_checkpoint(self, checkpoint_name: str) -> bool:
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            return False
        checkpoint = load_torch_checkpoint(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.loss_bundle.load_state_dict(checkpoint.get('loss_bundle_state_dict', {}))
        self.balancer.load_state_dict(checkpoint.get('balancer_state_dict', {}))
        return True

    def _compute_task_losses(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        outputs = self.model(batch['inputs'])
        task_losses = self.loss_bundle(outputs, batch)
        return outputs, task_losses

    @staticmethod
    def _inputs_to_canonical(inputs: torch.Tensor) -> torch.Tensor:
        # Flattened input convention is [B, xyzt=4, nu=2, flavor=3].
        if inputs.ndim != 2 or inputs.shape[-1] != 24:
            raise ValueError(f'Expected flattened F4 inputs [B,24], got {tuple(inputs.shape)}')
        return inputs.view(inputs.shape[0], 4, 2, 3).permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _canonical_to_flat(f4: torch.Tensor) -> torch.Tensor:
        # Canonical [B, nu, flavor, xyzt] -> flattened [B, xyzt, nu, flavor].
        return f4.permute(0, 3, 1, 2).contiguous().view(f4.shape[0], -1)

    def _compute_control_baseline(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.control_model is None:
            raise ValueError('Control baseline requested but control model is not configured.')

        f4 = self._inputs_to_canonical(inputs)
        if self.control_input_is_normalized:
            f4_norm = f4
            ntot = torch.ones((f4.shape[0],), device=f4.device, dtype=f4.dtype)
        else:
            ntot = torch.clamp(f4[:, :, :, 3].sum(dim=(1, 2)), min=1.0e-12)
            f4_norm = f4 / ntot[:, None, None, None]

        with torch.no_grad():
            box_f4_norm, box_growth_norm = self.control_model(f4_norm)

        finite_mask = torch.isfinite(box_f4_norm.reshape(box_f4_norm.shape[0], -1)).all(dim=1) & torch.isfinite(
            box_growth_norm
        )
        if not bool(torch.all(finite_mask)):
            box_f4_norm = torch.where(
                finite_mask[:, None, None, None],
                box_f4_norm,
                f4_norm,
            )
            box_growth_norm = torch.where(
                finite_mask,
                box_growth_norm,
                torch.zeros_like(box_growth_norm),
            )

        if self.control_input_is_normalized:
            box_f4 = box_f4_norm
            box_growth = box_growth_norm
        else:
            box_f4 = box_f4_norm * ntot[:, None, None, None]
            box_growth = box_growth_norm * ntot

        return self._canonical_to_flat(box_f4), box_growth.reshape(-1, 1)

    def _backward_standard(self, total_loss: torch.Tensor) -> None:
        if self.amp_enabled:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

    def _autocast_context(self):
        if self._use_new_amp:
            return torch.amp.autocast(device_type='cuda', enabled=self.amp_enabled)
        return torch.cuda.amp.autocast(enabled=self.amp_enabled)

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

        with self._autocast_context():
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

    def _build_accumulator(self, control_enabled: bool) -> EpochAccumulator:
        return EpochAccumulator(
            bc_threshold=self.bc_threshold,
            control_enabled=control_enabled,
            control_ratio_eps=self.control_ratio_eps,
            control_ratio_floor_quantile=self.control_ratio_floor_quantile,
            control_min_baseline_error=self.control_min_baseline_error,
        )

    def _run_loader(self, loader: Any, training: bool, epoch: int) -> dict[str, float]:
        if loader is None:
            return {}
        if training:
            self.model.train(True)
        else:
            self.model.train(False)

        prefix = 'train' if training else 'val'
        collect_control_metrics = self.control_enabled and (not training) and self.control_compute_during_fit
        accumulator = self._build_accumulator(control_enabled=collect_control_metrics)
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
            if collect_control_metrics:
                control_vector, control_reg = self._compute_control_baseline(batch['inputs'])
                accumulator.update_control(outputs, batch, control_vector, control_reg)

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

        restored_best = self._restore_checkpoint('best.pt')
        if restored_best and self.val_loader is not None:
            final_metrics.update(self.evaluate(self.val_loader, split_name='best_val'))

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
        accumulator = self._build_accumulator(control_enabled=self.control_enabled)
        progress = tqdm(loader, desc=split_name, leave=False)
        for raw_batch in progress:
            batch = move_batch_to_device(raw_batch, self.device)
            outputs, detached_log = self._evaluate_batch(batch)
            batch_size = int(batch['inputs'].shape[0])
            accumulator.update_losses(detached_log, batch_size)
            accumulator.update_outputs(outputs, batch)
            if self.control_enabled:
                control_vector, control_reg = self._compute_control_baseline(batch['inputs'])
                accumulator.update_control(outputs, batch, control_vector, control_reg)
        return accumulator.summarize(split_name)
