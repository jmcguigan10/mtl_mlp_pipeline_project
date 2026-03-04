from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class BaseLossBalancer(nn.Module):
    def __init__(self, task_specs: dict[str, Any]) -> None:
        super().__init__()
        self.task_names = list(task_specs.keys())
        self.task_kinds = {name: str(spec.kind) for name, spec in task_specs.items()}

    def aggregate(self, task_losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    def current_weight_dict(self) -> dict[str, float]:
        raise NotImplementedError


class StaticLossBalancer(BaseLossBalancer):
    def __init__(self, task_specs: dict[str, Any], weights: dict[str, float]) -> None:
        super().__init__(task_specs)
        self.weights = {name: float(weights.get(name, 1.0)) for name in self.task_names}

    def aggregate(self, task_losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        total = sum(self.weights[name] * task_losses[name] for name in self.task_names)
        return total, self.current_weight_dict()

    def weighted_losses(self, task_losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: self.weights[name] * task_losses[name] for name in self.task_names}

    def current_weight_dict(self) -> dict[str, float]:
        return dict(self.weights)


class KendallGalLossBalancer(BaseLossBalancer):
    def __init__(self, task_specs: dict[str, Any], initial_log_vars: dict[str, float]) -> None:
        super().__init__(task_specs)
        initial = [float(initial_log_vars.get(name, 0.0)) for name in self.task_names]
        self.log_vars = nn.Parameter(torch.tensor(initial, dtype=torch.float32))

    def aggregate(self, task_losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        total = torch.zeros((), device=self.log_vars.device)
        weight_info: dict[str, float] = {}
        for index, name in enumerate(self.task_names):
            log_var = self.log_vars[index]
            precision = torch.exp(-log_var)
            kind = self.task_kinds[name]
            if kind in {'regression', 'vector_regression'}:
                scaled = 0.5 * precision * task_losses[name] + 0.5 * log_var
                weight_info[name] = float((0.5 * precision).detach().item())
            else:
                scaled = precision * task_losses[name] + 0.5 * log_var
                weight_info[name] = float(precision.detach().item())
            total = total + scaled
        return total, weight_info

    def current_weight_dict(self) -> dict[str, float]:
        weights: dict[str, float] = {}
        for index, name in enumerate(self.task_names):
            precision = torch.exp(-self.log_vars[index])
            if self.task_kinds[name] in {'regression', 'vector_regression'}:
                weights[name] = float((0.5 * precision).detach().item())
            else:
                weights[name] = float(precision.detach().item())
        return weights


class GradNormLossBalancer(BaseLossBalancer):
    def __init__(self, task_specs: dict[str, Any], alpha: float, initial_weights: dict[str, float]) -> None:
        super().__init__(task_specs)
        initial = [float(initial_weights.get(name, 1.0)) for name in self.task_names]
        self.alpha = float(alpha)
        self.weights = nn.Parameter(torch.tensor(initial, dtype=torch.float32))
        self.register_buffer('initial_losses', torch.full((len(self.task_names),), float('nan')))

    def _positive_weights(self) -> torch.Tensor:
        clipped = torch.clamp(self.weights, min=1e-3)
        return len(self.task_names) * clipped / clipped.sum().clamp_min(1e-8)

    def aggregate(self, task_losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        weights = self._positive_weights().detach()
        total = torch.zeros((), device=weights.device)
        for index, name in enumerate(self.task_names):
            total = total + weights[index] * task_losses[name]
        return total, {name: float(weights[index].item()) for index, name in enumerate(self.task_names)}

    def current_weight_dict(self) -> dict[str, float]:
        weights = self._positive_weights().detach().cpu()
        return {name: float(weights[index].item()) for index, name in enumerate(self.task_names)}

    def compute_weight_gradients(
        self,
        task_losses: dict[str, torch.Tensor],
        shared_parameters: list[nn.Parameter],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        params = [parameter for parameter in shared_parameters if parameter.requires_grad]
        if not params:
            raise RuntimeError('GradNorm needs at least one shared trainable parameter.')

        if torch.isnan(self.initial_losses).any():
            with torch.no_grad():
                self.initial_losses.copy_(torch.tensor([task_losses[name].detach().item() for name in self.task_names], device=self.weights.device))

        weights = self._positive_weights()
        weighted_losses = [weights[index] * task_losses[name] for index, name in enumerate(self.task_names)]

        grad_norms: list[torch.Tensor] = []
        for index, weighted_loss in enumerate(weighted_losses):
            grads = torch.autograd.grad(
                weighted_loss,
                params,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )
            flattened = [gradient.reshape(-1) for gradient in grads if gradient is not None]
            if not flattened:
                grad_norm = torch.zeros((), device=self.weights.device)
            else:
                grad_norm = torch.norm(torch.cat(flattened), p=2)
            grad_norms.append(grad_norm)

        grad_norm_tensor = torch.stack(grad_norms)
        detached_losses = torch.tensor(
            [task_losses[name].detach().item() for name in self.task_names],
            device=self.weights.device,
            dtype=torch.float32,
        )
        loss_ratios = detached_losses / self.initial_losses.clamp_min(1e-8)
        inverse_train_rates = loss_ratios / loss_ratios.mean().clamp_min(1e-8)
        target = (grad_norm_tensor.detach().mean() * (inverse_train_rates ** self.alpha)).detach()
        gradnorm_loss = torch.abs(grad_norm_tensor - target).sum()
        weight_grads = torch.autograd.grad(gradnorm_loss, [self.weights], retain_graph=False)[0]

        stats = {f'gradnorm/{name}': float(grad_norm_tensor[index].detach().item()) for index, name in enumerate(self.task_names)}
        return weight_grads.detach(), stats

    def renormalize_(self) -> None:
        with torch.no_grad():
            self.weights.data.clamp_(min=1e-3)
            self.weights.data.mul_(len(self.task_names) / self.weights.data.sum().clamp_min(1e-8))



def build_loss_balancer(config: Any, task_specs: dict[str, Any]) -> BaseLossBalancer:
    name = str(config.multitask.loss_balancer.name).lower()
    if name == 'static':
        return StaticLossBalancer(task_specs, weights=dict(config.multitask.loss_balancer.get('static_weights', {})))
    if name == 'kendall_gal':
        return KendallGalLossBalancer(
            task_specs,
            initial_log_vars=dict(config.multitask.loss_balancer.kendall_gal.get('initial_log_vars', {})),
        )
    if name == 'gradnorm':
        gradnorm_cfg = config.multitask.loss_balancer.gradnorm
        return GradNormLossBalancer(
            task_specs,
            alpha=float(gradnorm_cfg.get('alpha', 1.5)),
            initial_weights=dict(gradnorm_cfg.get('initial_weights', {})),
        )
    raise ValueError(f'Unsupported loss balancer: {name}')
