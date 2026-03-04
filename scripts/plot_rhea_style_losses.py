from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _as_float_series(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        raw = row.get(key, '')
        if raw is None or raw == '':
            values.append(np.nan)
            continue
        try:
            values.append(float(raw))
        except ValueError:
            values.append(np.nan)
    return np.asarray(values, dtype=np.float64)


def _load_history(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f'No rows in history CSV: {path}')

    epoch = _as_float_series(rows, 'epoch')
    if np.all(~np.isfinite(epoch)):
        epoch = np.arange(1, len(rows) + 1, dtype=np.float64)

    series = {
        'train_bc': _as_float_series(rows, 'train/bc'),
        'eval_bc': _as_float_series(rows, 'val/bc'),
        'train_vec': _as_float_series(rows, 'train/vector_regression'),
        'eval_vec': _as_float_series(rows, 'val/vector_regression'),
        'train_reg': _as_float_series(rows, 'train/regression'),
        'eval_reg': _as_float_series(rows, 'val/regression'),
    }
    return epoch, series


def _plot_pair(
    epoch: np.ndarray,
    train_y: np.ndarray,
    eval_y: np.ndarray,
    title: str,
    out_path: Path,
    eval_label: str,
) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    ax.plot(epoch, train_y, label='train', linewidth=2.8)
    ax.plot(epoch, eval_y, label=eval_label, linewidth=2.8)
    ax.set_title(title, fontsize=30)
    ax.set_xlabel('epoch', fontsize=26)
    ax.set_ylabel('loss', fontsize=26)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(alpha=0.45)
    ax.legend(fontsize=24)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate Rhea-style loss plots from training history.csv')
    parser.add_argument(
        '--history',
        default='outputs/rhea_box3d_abs_train/history.csv',
        help='Path to training history.csv',
    )
    parser.add_argument(
        '--output_dir',
        default='plots',
        help='Directory to write PNG plots',
    )
    parser.add_argument(
        '--eval_label',
        default='test',
        help='Legend label for non-train curve (default keeps Rhea naming)',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    history_path = Path(args.history).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not history_path.exists():
        raise FileNotFoundError(f'History file not found: {history_path}')

    epoch, series = _load_history(history_path)

    _plot_pair(
        epoch=epoch,
        train_y=series['train_vec'],
        eval_y=series['eval_vec'],
        title='F4 loss',
        out_path=output_dir / 'f4_loss.png',
        eval_label=str(args.eval_label),
    )
    _plot_pair(
        epoch=epoch,
        train_y=series['train_reg'],
        eval_y=series['eval_reg'],
        title='Growth rate loss',
        out_path=output_dir / 'growthrate_loss.png',
        eval_label=str(args.eval_label),
    )
    _plot_pair(
        epoch=epoch,
        train_y=series['train_bc'],
        eval_y=series['eval_bc'],
        title='FFI (stability) loss',
        out_path=output_dir / 'ffi_loss.png',
        eval_label=str(args.eval_label),
    )
    print(f'Wrote plots to {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
