from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .blocks import MLPStack


class BaseTaskHead(nn.Module):
    def __init__(self, input_dim: int, config: Any, task_name: str) -> None:
        super().__init__()
        self.task_name = task_name
        self.mlp = MLPStack(input_dim, config)
        self.output_dim = int(config.output_dim)
        self.output_layer = nn.Linear(self.mlp.out_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return self.output_layer(x)


class BinaryClassificationHead(BaseTaskHead):
    pass


class VectorRegressionHead(BaseTaskHead):
    pass


class ScalarRegressionHead(BaseTaskHead):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)
