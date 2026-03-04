from __future__ import annotations

import random
from typing import Iterable

import torch


class PCGrad:
    def __init__(self, reduction: str = 'mean') -> None:
        reduction = reduction.lower()
        if reduction not in {'mean', 'sum'}:
            raise ValueError("PCGrad reduction must be 'mean' or 'sum'.")
        self.reduction = reduction

    @staticmethod
    def _dot(task_a: list[torch.Tensor], task_b: list[torch.Tensor]) -> torch.Tensor:
        return sum(torch.sum(a * b) for a, b in zip(task_a, task_b))

    @staticmethod
    def _norm_sq(task: list[torch.Tensor]) -> torch.Tensor:
        return sum(torch.sum(component * component) for component in task).clamp_min(1e-12)

    def pc_backward(self, objectives: list[torch.Tensor], parameters: Iterable[torch.nn.Parameter]) -> None:
        params = [parameter for parameter in parameters if parameter.requires_grad]
        if not objectives:
            raise ValueError('PCGrad requires at least one objective.')
        per_task_grads: list[list[torch.Tensor]] = []
        for objective_index, objective in enumerate(objectives):
            grads = torch.autograd.grad(
                objective,
                params,
                retain_graph=objective_index < len(objectives) - 1,
                allow_unused=True,
            )
            per_task_grads.append([
                torch.zeros_like(parameter) if gradient is None else gradient.detach().clone()
                for gradient, parameter in zip(grads, params)
            ])

        projected: list[list[torch.Tensor]] = []
        num_tasks = len(per_task_grads)
        for task_index in range(num_tasks):
            current = [component.clone() for component in per_task_grads[task_index]]
            order = list(range(num_tasks))
            random.shuffle(order)
            for other_index in order:
                if other_index == task_index:
                    continue
                other = per_task_grads[other_index]
                dot = self._dot(current, other)
                if dot < 0:
                    coeff = dot / self._norm_sq(other)
                    current = [component - coeff * other_component for component, other_component in zip(current, other)]
            projected.append(current)

        if self.reduction == 'mean':
            merged = [sum(task[param_index] for task in projected) / float(num_tasks) for param_index in range(len(params))]
        else:
            merged = [sum(task[param_index] for task in projected) for param_index in range(len(params))]

        for parameter, gradient in zip(params, merged):
            parameter.grad = gradient
