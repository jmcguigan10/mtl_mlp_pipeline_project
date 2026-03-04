from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..utils.common import module_from_path
from .balancers import GradNormLossBalancer, KendallGalLossBalancer


@dataclass
class SchedulerBundle:
    scheduler: Any | None
    step_on: str | None



def _make_optimizer(name: str, params: list[dict[str, Any]], defaults: dict[str, Any]) -> torch.optim.Optimizer:
    normalized = name.lower()
    if normalized == 'adamw':
        return torch.optim.AdamW(params, **defaults)
    if normalized == 'adam':
        return torch.optim.Adam(params, **defaults)
    if normalized == 'sgd':
        return torch.optim.SGD(params, momentum=float(defaults.pop('momentum', 0.9)), **defaults)
    raise ValueError(f'Unsupported optimizer: {name}')



def build_optimizers(config: Any, model: torch.nn.Module, balancer: torch.nn.Module) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer | None]:
    optimizer_cfg = config.training.optimizer
    defaults: dict[str, Any] = {
        'lr': float(optimizer_cfg.get('lr', 1e-3)),
        'weight_decay': float(optimizer_cfg.get('weight_decay', 0.0)),
        'betas': tuple(float(x) for x in optimizer_cfg.get('betas', [0.9, 0.999])),
        'eps': float(optimizer_cfg.get('eps', 1e-8)),
    }
    if str(optimizer_cfg.name).lower() == 'sgd':
        defaults = {
            'lr': float(optimizer_cfg.get('lr', 1e-3)),
            'weight_decay': float(optimizer_cfg.get('weight_decay', 0.0)),
            'momentum': float(optimizer_cfg.get('momentum', 0.9)),
        }

    param_groups: list[dict[str, Any]] = []
    used_ids: set[int] = set()

    for override in optimizer_cfg.get('param_groups', []):
        module_path = str(override.module)
        module = module_from_path(model, module_path)
        params = [parameter for parameter in module.parameters() if parameter.requires_grad and id(parameter) not in used_ids]
        if not params:
            continue
        group = {'params': params}
        for key, value in override.items():
            if key == 'module':
                continue
            group[key] = value
        param_groups.append(group)
        used_ids.update(id(parameter) for parameter in params)

    remaining_params = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in used_ids]
    if remaining_params:
        param_groups.append({'params': remaining_params})

    if isinstance(balancer, KendallGalLossBalancer):
        param_groups.append({'params': list(balancer.parameters()), 'weight_decay': 0.0})

    model_optimizer = _make_optimizer(str(optimizer_cfg.name), param_groups, defaults)

    balancer_optimizer: torch.optim.Optimizer | None = None
    if isinstance(balancer, GradNormLossBalancer):
        lr = float(config.multitask.loss_balancer.gradnorm.get('lr', 0.025))
        balancer_optimizer = torch.optim.Adam([balancer.weights], lr=lr)

    return model_optimizer, balancer_optimizer



def build_scheduler(
    config: Any,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
) -> SchedulerBundle:
    scheduler_cfg = config.training.get('scheduler', {'name': 'none'})
    name = str(scheduler_cfg.get('name', 'none')).lower()
    if name == 'none':
        return SchedulerBundle(scheduler=None, step_on=None)
    if name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_cfg.get('T_max', max(int(config.training.epochs), 1))),
            eta_min=float(scheduler_cfg.get('eta_min', 0.0)),
        )
        return SchedulerBundle(scheduler=scheduler, step_on='epoch')
    if name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_cfg.get('step_size', 10)),
            gamma=float(scheduler_cfg.get('gamma', 0.1)),
        )
        return SchedulerBundle(scheduler=scheduler, step_on='epoch')
    if name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_cfg.get('mode', 'min')),
            factor=float(scheduler_cfg.get('factor', 0.5)),
            patience=int(scheduler_cfg.get('patience', 3)),
        )
        return SchedulerBundle(scheduler=scheduler, step_on='metric')
    if name == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(scheduler_cfg.get('max_lr', config.training.optimizer.get('lr', 1e-3))),
            epochs=int(config.training.epochs),
            steps_per_epoch=max(steps_per_epoch, 1),
            pct_start=float(scheduler_cfg.get('pct_start', 0.3)),
        )
        return SchedulerBundle(scheduler=scheduler, step_on='batch')
    raise ValueError(f'Unsupported scheduler: {name}')
