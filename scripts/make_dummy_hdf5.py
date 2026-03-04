from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np



def _make_split(path: Path, num_samples: int, vector_dim: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(num_samples, 24)).astype(np.float32)

    bc_logits = 0.8 * inputs[:, 0] - 0.6 * inputs[:, 3] + 0.4 * inputs[:, 7] + rng.normal(scale=0.3, size=num_samples)
    bc = (bc_logits > 0).astype(np.float32).reshape(-1, 1)

    vector = np.stack(
        [
            0.5 * inputs[:, 1] + 0.2 * inputs[:, 2],
            -0.4 * inputs[:, 4] + 0.7 * inputs[:, 5],
            0.6 * inputs[:, 6] - 0.1 * inputs[:, 8],
        ],
        axis=1,
    ).astype(np.float32)
    if vector_dim > 3:
        extra = rng.normal(scale=0.2, size=(num_samples, vector_dim - 3)).astype(np.float32)
        vector = np.concatenate([vector, extra], axis=1)

    regression = (
        0.3 * inputs[:, 9]
        - 0.25 * inputs[:, 10]
        + 0.15 * inputs[:, 11]
        + 0.5 * bc.reshape(-1)
        + rng.normal(scale=0.2, size=num_samples)
    ).astype(np.float32).reshape(-1, 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('inputs', data=inputs)
        targets = handle.create_group('targets')
        targets.create_dataset('bc', data=bc)
        targets.create_dataset('vector', data=vector)
        targets.create_dataset('reg', data=regression)



def main() -> None:
    parser = argparse.ArgumentParser(description='Create synthetic HDF5 files for a smoke test.')
    parser.add_argument('--output_dir', required=True, help='Directory to place the generated files')
    parser.add_argument('--vector_dim', type=int, default=3, help='Dimension of the vector regression target')
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    _make_split(output_dir / 'train_a.h5', num_samples=512, vector_dim=args.vector_dim, seed=11)
    _make_split(output_dir / 'train_b.h5', num_samples=512, vector_dim=args.vector_dim, seed=17)
    _make_split(output_dir / 'val.h5', num_samples=256, vector_dim=args.vector_dim, seed=23)
    _make_split(output_dir / 'test.h5', num_samples=256, vector_dim=args.vector_dim, seed=31)
    print(f'Created synthetic HDF5 files in {output_dir}')


if __name__ == '__main__':
    main()
