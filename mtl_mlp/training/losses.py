from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseTaskLoss(nn.Module):
    def __init__(self, task_name: str, kind: str, loss_name: str) -> None:
        super().__init__()
        self.task_name = task_name
        self.kind = kind
        self.loss_name = loss_name

    @staticmethod
    def _apply_sample_weight(loss_tensor: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
        if sample_weight is None:
            return loss_tensor.mean()
        weight = sample_weight
        while weight.ndim < loss_tensor.ndim:
            weight = weight.unsqueeze(-1)
        weighted = loss_tensor * weight
        normalizer = weight.sum().clamp_min(1e-8)
        return weighted.sum() / normalizer


class BinaryClassificationLoss(BaseTaskLoss):
    def __init__(self, task_name: str, config: Any) -> None:
        super().__init__(task_name=task_name, kind=str(config.get('kind', 'binary_classification')), loss_name=str(config.name))
        pos_weight = config.get('pos_weight')
        if pos_weight is None:
            self.register_buffer('pos_weight_tensor', None, persistent=False)
        else:
            self.register_buffer('pos_weight_tensor', torch.tensor([float(pos_weight)], dtype=torch.float32), persistent=False)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.loss_name == 'bce_with_logits':
            logits = prediction.reshape_as(target)
            target = target.float()
            loss = F.binary_cross_entropy_with_logits(
                logits,
                target,
                reduction='none',
                pos_weight=self.pos_weight_tensor,
            )
            return self._apply_sample_weight(loss, sample_weight)

        if self.loss_name == 'cross_entropy':
            target = target.view(-1).long()
            loss = F.cross_entropy(prediction, target, reduction='none')
            return self._apply_sample_weight(loss, sample_weight)

        raise ValueError(f'Unsupported classification loss: {self.loss_name}')


class RegressionLoss(BaseTaskLoss):
    def __init__(self, task_name: str, kind: str, config: Any) -> None:
        super().__init__(task_name=task_name, kind=kind, loss_name=str(config.name))
        self.beta = float(config.get('beta', 1.0))

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target = target.float()
        if self.loss_name == 'mse':
            loss = F.mse_loss(prediction, target, reduction='none')
        elif self.loss_name == 'l1':
            loss = F.l1_loss(prediction, target, reduction='none')
        elif self.loss_name == 'smooth_l1':
            loss = F.smooth_l1_loss(prediction, target, beta=self.beta, reduction='none')
        else:
            raise ValueError(f'Unsupported regression loss: {self.loss_name}')
        return self._apply_sample_weight(loss, sample_weight)


@dataclass
class TaskSpec:
    name: str
    kind: str


class TaskLossBundle(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.losses = nn.ModuleDict(
            {
                'bc': BinaryClassificationLoss('bc', config.losses.bc),
                'vector_regression': RegressionLoss(
                    'vector_regression',
                    kind=str(config.losses.vector_regression.get('kind', 'vector_regression')),
                    config=config.losses.vector_regression,
                ),
                'regression': RegressionLoss(
                    'regression',
                    kind=str(config.losses.regression.get('kind', 'regression')),
                    config=config.losses.regression,
                ),
            }
        )
        self.task_specs = {
            'bc': TaskSpec(name='bc', kind=str(config.losses.bc.get('kind', 'binary_classification'))),
            'vector_regression': TaskSpec(
                name='vector_regression',
                kind=str(config.losses.vector_regression.get('kind', 'vector_regression')),
            ),
            'regression': TaskSpec(name='regression', kind=str(config.losses.regression.get('kind', 'regression'))),
        }

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sample_weight = batch.get('sample_weight')
        return {
            'bc': self.losses['bc'](outputs['bc'], batch['bc_target'], sample_weight=sample_weight),
            'vector_regression': self.losses['vector_regression'](
                outputs['vector_regression'], batch['vector_target'], sample_weight=sample_weight
            ),
            'regression': self.losses['regression'](
                outputs['regression'], batch['reg_target'], sample_weight=sample_weight
            ),
        }



def build_loss_bundle(config: Any) -> TaskLossBundle:
    return TaskLossBundle(config)
