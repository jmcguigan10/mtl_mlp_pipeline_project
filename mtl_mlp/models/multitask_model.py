from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..utils.common import freeze_module
from .blocks import MLPStack
from .equivariant_basis import EquivariantBasisTrunk
from .heads import BinaryClassificationHead, ScalarRegressionHead, VectorRegressionHead


class MultiTaskMLP(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.input_dim = int(config.model.input_dim)
        self.architecture = str(config.model.get('architecture', 'mlp')).lower()

        if self.architecture == 'mlp':
            self.trunk = MLPStack(self.input_dim, config.model.trunk)
            trunk_out_dim = self.trunk.out_dim
            self.heads = nn.ModuleDict(
                {
                    'bc': BinaryClassificationHead(trunk_out_dim, config.model.heads.bc, task_name='bc'),
                    'vector_regression': VectorRegressionHead(
                        trunk_out_dim,
                        config.model.heads.vector_regression,
                        task_name='vector_regression',
                    ),
                    'regression': ScalarRegressionHead(
                        trunk_out_dim,
                        config.model.heads.regression,
                        task_name='regression',
                    ),
                }
            )
        elif self.architecture == 'equivariant_basis':
            self.trunk = EquivariantBasisTrunk(self.input_dim, config.model.trunk)
            vector_out_dim = int(config.model.heads.vector_regression.output_dim)
            if vector_out_dim != 24:
                raise ValueError(
                    'model.heads.vector_regression.output_dim must be 24 for architecture=equivariant_basis'
                )
            pooled_dim = self.trunk.pooled_dim
            self.heads = nn.ModuleDict(
                {
                    # Flux output is produced directly by equivariant basis mixing.
                    'vector_regression': nn.Identity(),
                    'bc': BinaryClassificationHead(pooled_dim, config.model.heads.bc, task_name='bc'),
                    'regression': ScalarRegressionHead(
                        pooled_dim,
                        config.model.heads.regression,
                        task_name='regression',
                    ),
                }
            )
        else:
            raise ValueError(f'Unsupported model architecture: {self.architecture}')

        self._apply_freeze_policy()

    def _apply_freeze_policy(self) -> None:
        if bool(self.config.model.trunk.get('freeze', False)):
            freeze_module(self.trunk)
        if bool(self.config.model.heads.bc.get('freeze', False)):
            freeze_module(self.heads['bc'])
        if bool(self.config.model.heads.vector_regression.get('freeze', False)):
            freeze_module(self.heads['vector_regression'])
        if bool(self.config.model.heads.regression.get('freeze', False)):
            freeze_module(self.heads['regression'])

    def train(self, mode: bool = True) -> 'MultiTaskMLP':
        super().train(mode)
        # Keep frozen submodules in eval mode, especially for BatchNorm.
        if bool(self.config.model.trunk.get('freeze', False)):
            self.trunk.eval()
        if bool(self.config.model.heads.bc.get('freeze', False)):
            self.heads['bc'].eval()
        if bool(self.config.model.heads.vector_regression.get('freeze', False)):
            self.heads['vector_regression'].eval()
        if bool(self.config.model.heads.regression.get('freeze', False)):
            self.heads['regression'].eval()
        return self

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError(f'Expected input tensor of shape [batch, {self.input_dim}], got {tuple(inputs.shape)}')
        if self.architecture == 'mlp':
            features = self.trunk(inputs)
            return {
                'bc': self.heads['bc'](features),
                'vector_regression': self.heads['vector_regression'](features),
                'regression': self.heads['regression'](features),
            }

        trunk_out = self.trunk(inputs)
        pooled = trunk_out['pooled']
        return {
            'bc': self.heads['bc'](pooled),
            'vector_regression': self.heads['vector_regression'](trunk_out['vector_flat']),
            'regression': self.heads['regression'](pooled),
        }

    def get_shared_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.trunk.parameters() if parameter.requires_grad]

    def parameter_groups_for_overrides(self) -> dict[str, list[nn.Parameter]]:
        groups: dict[str, list[nn.Parameter]] = {}
        groups['trunk'] = [parameter for parameter in self.trunk.parameters() if parameter.requires_grad]
        for head_name, head in self.heads.items():
            groups[f'heads.{head_name}'] = [parameter for parameter in head.parameters() if parameter.requires_grad]
        return groups
