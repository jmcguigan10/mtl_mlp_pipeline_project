from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtl_mlp.config import load_config
from mtl_mlp.models import MultiTaskMLP

CONFIG = ROOT / 'configs' / 'rhea_equivariant_abs_smoke.yaml'


def flat_to_canonical(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.shape[0], 4, 2, 3).permute(0, 2, 3, 1).contiguous()


def canonical_to_flat(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2).contiguous().view(x.shape[0], -1)


def random_rotation_matrix() -> torch.Tensor:
    # Deterministic-ish random rotation from QR decomposition.
    a = torch.randn(3, 3, dtype=torch.float32)
    q, r = torch.linalg.qr(a)
    d = torch.diag(r)
    s = torch.sign(d)
    s = torch.where(s == 0, torch.ones_like(s), s)
    q = q * s
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def apply_rotation(x_flat: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    x = flat_to_canonical(x_flat)
    v = x[..., :3]
    n = x[..., 3:4]
    v_rot = torch.einsum('ij,bnfj->bnfi', rot, v)
    return canonical_to_flat(torch.cat([v_rot, n], dim=-1))


def apply_flavor_permutation(x_flat: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    x = flat_to_canonical(x_flat)
    x_perm = x[:, :, perm, :]
    return canonical_to_flat(x_perm)


def main() -> None:
    torch.manual_seed(7)
    config = load_config(str(CONFIG))
    model = MultiTaskMLP(config)
    model.eval()

    batch = 16
    x = torch.randn(batch, 24, dtype=torch.float32)
    rot = random_rotation_matrix()

    with torch.no_grad():
        y = model(x)
        x_rot = apply_rotation(x, rot)
        y_rot = model(x_rot)

    y_vec = flat_to_canonical(y['vector_regression'])
    y_vec_rot_expected = torch.cat([torch.einsum('ij,bnfj->bnfi', rot, y_vec[..., :3]), y_vec[..., 3:4]], dim=-1)
    y_vec_rot_pred = flat_to_canonical(y_rot['vector_regression'])

    vec_rot_max = float(torch.max(torch.abs(y_vec_rot_expected - y_vec_rot_pred)).item())
    reg_rot_max = float(torch.max(torch.abs(y['regression'] - y_rot['regression'])).item())
    bc_rot_max = float(torch.max(torch.abs(y['bc'] - y_rot['bc'])).item())

    perm = torch.tensor([2, 0, 1], dtype=torch.long)
    with torch.no_grad():
        x_perm = apply_flavor_permutation(x, perm)
        y_perm = model(x_perm)

    y_vec_perm_expected = flat_to_canonical(y['vector_regression'])[:, :, perm, :]
    y_vec_perm_pred = flat_to_canonical(y_perm['vector_regression'])
    vec_perm_max = float(torch.max(torch.abs(y_vec_perm_expected - y_vec_perm_pred)).item())
    reg_perm_max = float(torch.max(torch.abs(y['regression'] - y_perm['regression'])).item())
    bc_perm_max = float(torch.max(torch.abs(y['bc'] - y_perm['bc'])).item())

    tol = 1.0e-4
    if vec_rot_max > tol or reg_rot_max > tol or bc_rot_max > tol:
        raise ValueError(
            f'Rotation equivariance/invariance failed: vec={vec_rot_max:.3e}, reg={reg_rot_max:.3e}, bc={bc_rot_max:.3e}'
        )
    if vec_perm_max > tol or reg_perm_max > tol or bc_perm_max > tol:
        raise ValueError(
            f'Permutation equivariance/invariance failed: vec={vec_perm_max:.3e}, reg={reg_perm_max:.3e}, bc={bc_perm_max:.3e}'
        )

    print(
        'Equivariant-basis checks passed '
        f'(rot vec/reg/bc max={vec_rot_max:.3e}/{reg_rot_max:.3e}/{bc_rot_max:.3e}, '
        f'perm vec/reg/bc max={vec_perm_max:.3e}/{reg_perm_max:.3e}/{bc_perm_max:.3e})'
    )


if __name__ == '__main__':
    main()
