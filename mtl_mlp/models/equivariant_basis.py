from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from .blocks import MLPStack


def _as_mlp_config(config: Any, fallback: Any) -> Any:
    if config is None:
        return fallback
    return config


class EquivariantBasisTrunk(nn.Module):
    """SO(3)-equivariant basis mixing over [nu/nubar, flavor] tokens.

    Inputs are flattened [B, 24] tensors corresponding to [B, xyzt=4, nu=2, flavor=3].
    Vector outputs are built only from vector bases with scalar coefficients so equivariance
    is exact by construction.
    """

    def __init__(self, input_dim: int, trunk_config: Any) -> None:
        super().__init__()
        if int(input_dim) != 24:
            raise ValueError(f'EquivariantBasisTrunk expects input_dim=24, got {input_dim}')

        self.input_dim = int(input_dim)
        eq_cfg = trunk_config.get('equivariant', {})
        self.eps = float(eq_cfg.get('eps', 1.0e-8))

        # For the fixed 2x3 token layout: 4 n terms + 4 norms + 6 pairwise dots.
        self.local_invariant_dim = 14
        # Permutation-invariant pooled moments of local invariants: mean + second moment.
        self.global_invariant_dim = 2 * self.local_invariant_dim

        context_cfg = _as_mlp_config(eq_cfg.get('context_mlp'), trunk_config)
        token_cfg = _as_mlp_config(eq_cfg.get('token_mlp'), trunk_config)

        self.context_mlp = MLPStack(self.global_invariant_dim, context_cfg)
        self.token_mlp = MLPStack(self.local_invariant_dim + self.context_mlp.out_dim, token_cfg)
        self.vector_coeff = nn.Linear(self.token_mlp.out_dim, 4)
        self.scalar_coeff = nn.Linear(self.token_mlp.out_dim, 4)
        self.scalar_bias = nn.Linear(self.token_mlp.out_dim, 1)

        pooled_dim = int(eq_cfg.get('pooled_dim', self.token_mlp.out_dim))
        pooled_in = self.context_mlp.out_dim + self.local_invariant_dim + self.token_mlp.out_dim
        self.pooled_projection = nn.Linear(pooled_in, pooled_dim)
        self.pooled_dim = pooled_dim

    @staticmethod
    def _to_canonical(x: torch.Tensor) -> torch.Tensor:
        # [B, 4, 2, 3] -> [B, 2, 3, 4]
        return x.view(x.shape[0], 4, 2, 3).permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _to_flat(f4: torch.Tensor) -> torch.Tensor:
        # [B, 2, 3, 4] -> [B, 4, 2, 3] -> [B, 24]
        return f4.permute(0, 3, 1, 2).contiguous().view(f4.shape[0], -1)

    @staticmethod
    def _relation_means(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: [B, nu, flavor, C]
        x_self = x
        n_nu = x.shape[1]
        n_flavor = x.shape[2]

        x_flavor_raw = x.sum(dim=2, keepdim=True) - x_self
        x_nunubar_raw = x.sum(dim=1, keepdim=True) - x_self
        x_all_raw = x.sum(dim=(1, 2), keepdim=True) - x_self - x_flavor_raw - x_nunubar_raw

        flavor_denom = max(n_flavor - 1, 1)
        nunubar_denom = max(n_nu - 1, 1)
        all_denom = max((n_flavor - 1) * (n_nu - 1), 1)

        x_flavor = x_flavor_raw / float(flavor_denom)
        x_nunubar = x_nunubar_raw / float(nunubar_denom)
        x_all = x_all_raw / float(all_denom)
        return x_self, x_flavor, x_nunubar, x_all

    def _build_local_invariants(
        self,
        n_rel: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        v_rel: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        n_self, n_flavor, n_nunubar, n_all = n_rel
        v_self, v_flavor, v_nunubar, v_all = v_rel

        norm_self = torch.sqrt(torch.sum(v_self * v_self, dim=-1, keepdim=True) + self.eps)
        norm_flavor = torch.sqrt(torch.sum(v_flavor * v_flavor, dim=-1, keepdim=True) + self.eps)
        norm_nunubar = torch.sqrt(torch.sum(v_nunubar * v_nunubar, dim=-1, keepdim=True) + self.eps)
        norm_all = torch.sqrt(torch.sum(v_all * v_all, dim=-1, keepdim=True) + self.eps)

        dot_self_flavor = torch.sum(v_self * v_flavor, dim=-1, keepdim=True)
        dot_self_nunubar = torch.sum(v_self * v_nunubar, dim=-1, keepdim=True)
        dot_self_all = torch.sum(v_self * v_all, dim=-1, keepdim=True)
        dot_flavor_nunubar = torch.sum(v_flavor * v_nunubar, dim=-1, keepdim=True)
        dot_flavor_all = torch.sum(v_flavor * v_all, dim=-1, keepdim=True)
        dot_nunubar_all = torch.sum(v_nunubar * v_all, dim=-1, keepdim=True)

        return torch.cat(
            [
                n_self,
                n_flavor,
                n_nunubar,
                n_all,
                norm_self,
                norm_flavor,
                norm_nunubar,
                norm_all,
                dot_self_flavor,
                dot_self_nunubar,
                dot_self_all,
                dot_flavor_nunubar,
                dot_flavor_all,
                dot_nunubar_all,
            ],
            dim=-1,
        )

    @staticmethod
    def _build_global_invariants(local_invariants: torch.Tensor) -> torch.Tensor:
        mean = local_invariants.mean(dim=(1, 2))
        second_moment = torch.mean(local_invariants * local_invariants, dim=(1, 2))
        return torch.cat([mean, second_moment], dim=-1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError(f'Expected input tensor of shape [batch, {self.input_dim}], got {tuple(inputs.shape)}')

        x = self._to_canonical(inputs)
        v = x[..., :3]
        n = x[..., 3:4]

        n_rel = self._relation_means(n)
        v_rel = self._relation_means(v)
        local_invariants = self._build_local_invariants(n_rel, v_rel)
        global_invariants = self._build_global_invariants(local_invariants)

        context = self.context_mlp(global_invariants)
        context_expanded = context[:, None, None, :].expand(local_invariants.shape[0], 2, 3, context.shape[-1])
        token_in = torch.cat([local_invariants, context_expanded], dim=-1)
        token_hidden = self.token_mlp(token_in.view(-1, token_in.shape[-1])).view(token_in.shape[0], 2, 3, -1)

        vec_coeff = self.vector_coeff(token_hidden)
        scalar_coeff = self.scalar_coeff(token_hidden)
        scalar_bias = self.scalar_bias(token_hidden)

        v_self, v_flavor, v_nunubar, v_all = v_rel
        n_self, n_flavor, n_nunubar, n_all = n_rel
        v_out = (
            vec_coeff[..., 0:1] * v_self
            + vec_coeff[..., 1:2] * v_flavor
            + vec_coeff[..., 2:3] * v_nunubar
            + vec_coeff[..., 3:4] * v_all
        )
        n_out = (
            scalar_coeff[..., 0:1] * n_self
            + scalar_coeff[..., 1:2] * n_flavor
            + scalar_coeff[..., 2:3] * n_nunubar
            + scalar_coeff[..., 3:4] * n_all
            + scalar_bias
        )

        f4_out = torch.cat([v_out, n_out], dim=-1)
        vector_flat = self._to_flat(f4_out)

        pooled_input = torch.cat(
            [
                context,
                local_invariants.mean(dim=(1, 2)),
                token_hidden.mean(dim=(1, 2)),
            ],
            dim=-1,
        )
        pooled = self.pooled_projection(pooled_input)
        return {
            'vector_flat': vector_flat,
            'pooled': pooled,
        }
