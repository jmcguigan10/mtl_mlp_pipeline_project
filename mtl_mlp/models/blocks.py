from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class FeatureRecalibration(nn.Module):
    """Simple squeeze/excitation-style gating for vector features."""

    def __init__(self, dim: int, reduction: int = 4, min_hidden_dim: int = 4) -> None:
        super().__init__()
        hidden_dim = max(dim // max(reduction, 1), min_hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)



def build_activation(name: str) -> nn.Module:
    normalized = str(name).lower()
    if normalized == 'relu':
        return nn.ReLU(inplace=True)
    if normalized == 'gelu':
        return nn.GELU()
    if normalized == 'silu':
        return nn.SiLU(inplace=True)
    if normalized == 'elu':
        return nn.ELU(inplace=True)
    if normalized == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    if normalized == 'tanh':
        return nn.Tanh()
    if normalized == 'identity':
        return nn.Identity()
    raise ValueError(f'Unsupported activation: {name}')



def build_norm(dim: int, batch_norm: bool = False, layer_norm: bool = False) -> nn.Module:
    if batch_norm and layer_norm:
        raise ValueError('Use either batch_norm or layer_norm for a block, not both.')
    if batch_norm:
        return nn.BatchNorm1d(dim)
    if layer_norm:
        return nn.LayerNorm(dim)
    return nn.Identity()


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = 'relu',
        batch_norm: bool = False,
        layer_norm: bool = False,
        dropout: float = 0.0,
        residual: bool = False,
        recalibration: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        recalibration = recalibration or {'enabled': False}
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = build_norm(out_dim, batch_norm=batch_norm, layer_norm=layer_norm)
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.recalibration = (
            FeatureRecalibration(
                dim=out_dim,
                reduction=int(recalibration.get('reduction', 4)),
                min_hidden_dim=int(recalibration.get('min_hidden_dim', 4)),
            )
            if recalibration.get('enabled', False)
            else nn.Identity()
        )
        self.use_residual = bool(residual) and in_dim == out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.recalibration(x)
        if self.use_residual:
            x = x + residual
        return x


class MLPStack(nn.Module):
    def __init__(self, in_dim: int, config: Any) -> None:
        super().__init__()
        hidden_dims = [int(dim) for dim in config.get('hidden_dims', [])]
        activation = str(config.get('activation', 'relu'))
        batch_norm = bool(config.get('batch_norm', False))
        layer_norm = bool(config.get('layer_norm', False))
        dropout = float(config.get('dropout', 0.0))
        residual = bool(config.get('residual', False))
        recalibration = config.get('recalibration', {'enabled': False})

        blocks: list[nn.Module] = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            blocks.append(
                MLPBlock(
                    in_dim=prev_dim,
                    out_dim=hidden_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    layer_norm=layer_norm,
                    dropout=dropout,
                    residual=residual,
                    recalibration=recalibration,
                )
            )
            prev_dim = hidden_dim
        self.network = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.out_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
