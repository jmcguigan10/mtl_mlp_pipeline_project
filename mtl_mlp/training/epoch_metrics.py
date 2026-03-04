from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class EpochAccumulator:
    bc_threshold: float = 0.5
    control_enabled: bool = False
    control_ratio_eps: float = 1.0e-8
    control_ratio_floor_quantile: float = 0.10
    control_min_baseline_error: float = 1.0e-4

    def __post_init__(self) -> None:
        self.loss_sums: dict[str, float] = {}
        self.num_samples = 0
        self.bc_correct = 0.0
        self.bc_weight_total = 0.0
        self.bc_tp = 0.0
        self.bc_fp = 0.0
        self.bc_fn = 0.0
        self.bc_tn = 0.0
        self.reg_abs = 0.0
        self.reg_sq = 0.0
        self.reg_count = 0.0
        self.vec_abs = 0.0
        self.vec_sq = 0.0
        self.vec_l2 = 0.0
        self.vec_count = 0.0
        self.vec_items = 0.0
        self.control_vec_abs = 0.0
        self.control_reg_abs = 0.0
        self.control_vec_count = 0.0
        self.control_reg_count = 0.0
        self.vector_control_ratios: list[torch.Tensor] = []
        self.reg_control_ratios: list[torch.Tensor] = []
        self.vector_control_wins = 0.0
        self.reg_control_wins = 0.0
        self.vector_control_valid = 0.0
        self.reg_control_valid = 0.0
        self.vector_control_skipped = 0.0
        self.reg_control_skipped = 0.0

    def update_losses(self, losses: dict[str, float], batch_size: int) -> None:
        self.num_samples += batch_size
        for key, value in losses.items():
            self.loss_sums[key] = self.loss_sums.get(key, 0.0) + float(value) * batch_size

    @staticmethod
    def _task_weight(batch: dict[str, torch.Tensor], key: str, reference: torch.Tensor) -> torch.Tensor:
        batch_size = int(reference.shape[0])
        raw = batch.get(key)
        if raw is None:
            raw = batch.get('sample_weight')
        if raw is None:
            return torch.ones((batch_size,), device=reference.device, dtype=reference.dtype)
        w = raw.detach().to(device=reference.device, dtype=reference.dtype)
        if w.ndim == 0:
            return w.repeat(batch_size).clamp_min(0.0)
        if w.ndim == 1 and w.shape[0] == batch_size:
            return w.clamp_min(0.0)
        return w.reshape(batch_size, -1).mean(dim=1).clamp_min(0.0)

    def update_control(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        control_vector_target: torch.Tensor,
        control_reg_target: torch.Tensor,
    ) -> None:
        if not self.control_enabled:
            return

        vec_pred = outputs['vector_regression'].detach()
        vec_target = batch['vector_target'].detach()
        vec_control = control_vector_target.detach()
        vec_weight = self._task_weight(batch, 'vector_sample_weight', vec_pred)

        reg_pred = outputs['regression'].detach()
        reg_target = batch['reg_target'].detach()
        reg_control = control_reg_target.detach()
        reg_weight = self._task_weight(batch, 'reg_sample_weight', reg_pred)

        batch_size = int(vec_pred.shape[0])
        vec_err = torch.mean(torch.abs(vec_pred - vec_target).reshape(batch_size, -1), dim=1)
        vec_control_err = torch.mean(torch.abs(vec_control - vec_target).reshape(batch_size, -1), dim=1)
        reg_err = torch.mean(torch.abs(reg_pred - reg_target).reshape(batch_size, -1), dim=1)
        reg_control_err = torch.mean(torch.abs(reg_control - reg_target).reshape(batch_size, -1), dim=1)

        vec_mask = vec_weight > 0.0
        reg_mask = reg_weight > 0.0

        if bool(torch.any(vec_mask)):
            vec_err = vec_err[vec_mask]
            vec_control_err = vec_control_err[vec_mask]
            vec_min_baseline = max(self.control_ratio_eps, self.control_min_baseline_error)
            vec_valid = vec_control_err > vec_min_baseline
            self.vector_control_skipped += float((~vec_valid).sum().item())
            if bool(torch.any(vec_valid)):
                vec_err = vec_err[vec_valid]
                vec_control_err = vec_control_err[vec_valid]
                vec_floor = torch.clamp(
                    torch.quantile(vec_control_err, min(max(self.control_ratio_floor_quantile, 0.0), 1.0)),
                    min=vec_min_baseline,
                )
                vec_ratio = torch.abs(vec_err / torch.clamp(vec_control_err, min=vec_floor))
                self.control_vec_abs += float(vec_control_err.sum().item())
                self.control_vec_count += float(vec_err.numel())
                self.vector_control_wins += float((vec_err < vec_control_err).sum().item())
                self.vector_control_valid += float(vec_err.numel())
                self.vector_control_ratios.append(vec_ratio.detach().cpu())

        if bool(torch.any(reg_mask)):
            reg_err = reg_err[reg_mask]
            reg_control_err = reg_control_err[reg_mask]
            reg_min_baseline = max(self.control_ratio_eps, self.control_min_baseline_error)
            reg_valid = reg_control_err > reg_min_baseline
            self.reg_control_skipped += float((~reg_valid).sum().item())
            if bool(torch.any(reg_valid)):
                reg_err = reg_err[reg_valid]
                reg_control_err = reg_control_err[reg_valid]
                reg_floor = torch.clamp(
                    torch.quantile(reg_control_err, min(max(self.control_ratio_floor_quantile, 0.0), 1.0)),
                    min=reg_min_baseline,
                )
                reg_ratio = torch.abs(reg_err / torch.clamp(reg_control_err, min=reg_floor))
                self.control_reg_abs += float(reg_control_err.sum().item())
                self.control_reg_count += float(reg_err.numel())
                self.reg_control_wins += float((reg_err < reg_control_err).sum().item())
                self.reg_control_valid += float(reg_err.numel())
                self.reg_control_ratios.append(reg_ratio.detach().cpu())

    @staticmethod
    def _summarize_ratio(values: list[torch.Tensor]) -> dict[str, float]:
        if not values:
            return {
                'mean': float('nan'),
                'median': float('nan'),
                'p95': float('nan'),
                'p99': float('nan'),
            }
        flat = torch.cat(values, dim=0)
        finite = flat[torch.isfinite(flat)]
        if finite.numel() == 0:
            return {
                'mean': float('nan'),
                'median': float('nan'),
                'p95': float('nan'),
                'p99': float('nan'),
            }
        return {
            'mean': float(torch.mean(finite).item()),
            'median': float(torch.quantile(finite, 0.5).item()),
            'p95': float(torch.quantile(finite, 0.95).item()),
            'p99': float(torch.quantile(finite, 0.99).item()),
        }

    def update_outputs(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        bc_logits = outputs['bc'].detach()
        bc_target = batch['bc_target'].detach()
        bc_ref = bc_logits.reshape(bc_logits.shape[0], -1)
        bc_weight = self._task_weight(batch, 'bc_sample_weight', bc_ref)
        bc_valid = bc_weight > 0.0

        if bc_logits.shape[-1] == 1:
            probs = torch.sigmoid(bc_logits.reshape_as(bc_target)).reshape(bc_target.shape[0], -1)[:, 0]
            targets = bc_target.reshape(bc_target.shape[0], -1)[:, 0].long()
            if bool(torch.any(bc_valid)):
                preds = (probs >= self.bc_threshold).long()[bc_valid]
                targets = targets[bc_valid]
                weights = bc_weight[bc_valid]
                self.bc_correct += float(((preds == targets).float() * weights).sum().item())
                self.bc_weight_total += float(weights.sum().item())
                self.bc_tp += float((((preds == 1) & (targets == 1)).float() * weights).sum().item())
                self.bc_fp += float((((preds == 1) & (targets == 0)).float() * weights).sum().item())
                self.bc_fn += float((((preds == 0) & (targets == 1)).float() * weights).sum().item())
                self.bc_tn += float((((preds == 0) & (targets == 0)).float() * weights).sum().item())
        else:
            preds = torch.argmax(bc_logits, dim=-1)
            targets = bc_target.view(-1).long()
            if bool(torch.any(bc_valid)):
                preds = preds[bc_valid]
                targets = targets[bc_valid]
                weights = bc_weight[bc_valid]
                self.bc_correct += float(((preds == targets).float() * weights).sum().item())
                self.bc_weight_total += float(weights.sum().item())

        reg_pred = outputs['regression'].detach()
        reg_target = batch['reg_target'].detach()
        reg_diff = reg_pred - reg_target
        reg_weight = self._task_weight(batch, 'reg_sample_weight', reg_diff)
        reg_diff_flat = reg_diff.reshape(reg_diff.shape[0], -1)
        reg_abs = torch.abs(reg_diff_flat)
        reg_sq = reg_diff_flat ** 2
        self.reg_abs += float((reg_abs * reg_weight[:, None]).sum().item())
        self.reg_sq += float((reg_sq * reg_weight[:, None]).sum().item())
        self.reg_count += float((reg_weight.sum() * reg_diff_flat.shape[1]).item())

        vec_pred = outputs['vector_regression'].detach()
        vec_target = batch['vector_target'].detach()
        vec_diff = vec_pred - vec_target
        vec_weight = self._task_weight(batch, 'vector_sample_weight', vec_diff)
        vec_diff_flat = vec_diff.reshape(vec_diff.shape[0], -1)
        vec_abs = torch.abs(vec_diff_flat)
        vec_sq = vec_diff_flat ** 2
        self.vec_abs += float((vec_abs * vec_weight[:, None]).sum().item())
        self.vec_sq += float((vec_sq * vec_weight[:, None]).sum().item())
        self.vec_l2 += float((torch.norm(vec_diff_flat, dim=-1) * vec_weight).sum().item())
        self.vec_count += float(vec_weight.sum().item())
        self.vec_items += float((vec_weight.sum() * vec_diff_flat.shape[1]).item())

    def summarize(self, prefix: str) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for key, value in self.loss_sums.items():
            metrics[f'{prefix}/{key}'] = value / max(self.num_samples, 1)
        if self.bc_weight_total > 0:
            metrics[f'{prefix}/bc_accuracy'] = self.bc_correct / max(self.bc_weight_total, 1.0)
        total_bc = self.bc_tp + self.bc_fp + self.bc_fn + self.bc_tn
        if total_bc > 0:
            precision = self.bc_tp / max(self.bc_tp + self.bc_fp, 1.0)
            recall = self.bc_tp / max(self.bc_tp + self.bc_fn, 1.0)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
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
        if self.control_enabled:
            vec_stats = self._summarize_ratio(self.vector_control_ratios)
            reg_stats = self._summarize_ratio(self.reg_control_ratios)
            metrics[f'{prefix}/control_vector_mae'] = self.control_vec_abs / max(self.control_vec_count, 1)
            metrics[f'{prefix}/control_reg_mae'] = self.control_reg_abs / max(self.control_reg_count, 1)
            metrics[f'{prefix}/control_growth_mae'] = metrics[f'{prefix}/control_reg_mae']
            metrics[f'{prefix}/vector_vs_control_win_rate'] = self.vector_control_wins / max(self.vector_control_valid, 1.0)
            metrics[f'{prefix}/reg_vs_control_win_rate'] = self.reg_control_wins / max(self.reg_control_valid, 1.0)
            metrics[f'{prefix}/vector_vs_control_skipped_frac'] = self.vector_control_skipped / max(
                self.vector_control_valid + self.vector_control_skipped,
                1.0,
            )
            metrics[f'{prefix}/reg_vs_control_skipped_frac'] = self.reg_control_skipped / max(
                self.reg_control_valid + self.reg_control_skipped,
                1.0,
            )
            metrics[f'{prefix}/vector_vs_control_frac_mean'] = vec_stats['mean']
            metrics[f'{prefix}/vector_vs_control_frac_median'] = vec_stats['median']
            metrics[f'{prefix}/vector_vs_control_frac_p95'] = vec_stats['p95']
            metrics[f'{prefix}/vector_vs_control_frac_p99'] = vec_stats['p99']
            metrics[f'{prefix}/reg_vs_control_frac_mean'] = reg_stats['mean']
            metrics[f'{prefix}/reg_vs_control_frac_median'] = reg_stats['median']
            metrics[f'{prefix}/reg_vs_control_frac_p95'] = reg_stats['p95']
            metrics[f'{prefix}/reg_vs_control_frac_p99'] = reg_stats['p99']
            metrics[f'{prefix}/growth_vs_control_frac_mean'] = reg_stats['mean']
            metrics[f'{prefix}/growth_vs_control_frac_median'] = reg_stats['median']
            metrics[f'{prefix}/growth_vs_control_frac_p95'] = reg_stats['p95']
            metrics[f'{prefix}/growth_vs_control_frac_p99'] = reg_stats['p99']
            metrics[f'{prefix}/vector_error_vs_control_pct'] = 100.0 * vec_stats['mean']
            metrics[f'{prefix}/reg_error_vs_control_pct'] = 100.0 * reg_stats['mean']
            metrics[f'{prefix}/growth_error_vs_control_pct'] = 100.0 * reg_stats['mean']
        return metrics
