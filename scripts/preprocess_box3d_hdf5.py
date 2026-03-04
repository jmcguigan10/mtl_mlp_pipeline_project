from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtl_mlp.preprocessing.ambiguity_filter import prepare_ambiguity_weights


F4_INITIAL_KEY = "F4_initial(1|ccm)"
F4_FINAL_KEY = "F4_final(1|ccm)"
STABLE_KEY = "stable"
GROWTH_KEY = "growthRate(1|s)"
TARGET_GROUP = "targets"
TARGET_F4_KEY = f"{TARGET_GROUP}/{F4_FINAL_KEY}"
TARGET_GROWTH_KEY = f"{TARGET_GROUP}/{GROWTH_KEY}"
MASK_GROUP = "masks"
MASK_BC_WEIGHT_KEY = f"{MASK_GROUP}/bc_target_weight"
MASK_VECTOR_WEIGHT_KEY = f"{MASK_GROUP}/vector_target_weight"
MASK_REG_WEIGHT_KEY = f"{MASK_GROUP}/reg_target_weight"
NORM_PREFIX = "normalized"
NORM_F4_INITIAL_KEY = f"{NORM_PREFIX}/{F4_INITIAL_KEY}"
NORM_TARGET_F4_KEY = f"{NORM_PREFIX}/{TARGET_F4_KEY}"
NORM_TARGET_GROWTH_KEY = f"{NORM_PREFIX}/{TARGET_GROWTH_KEY}"


@dataclass(frozen=True)
class LayoutSpec:
    axis_nu: int
    axis_flavor: int
    axis_xyzt: int


def _as_float32(value: np.ndarray | h5py.Dataset) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def infer_layout(shape: tuple[int, ...]) -> LayoutSpec:
    if len(shape) != 4:
        raise ValueError(f"Expected {F4_INITIAL_KEY} shape [N, *, *, *], got {shape}")
    dims = shape[1:]

    xyzt_axes = [i for i, dim in enumerate(dims) if dim == 4]
    nu_axes = [i for i, dim in enumerate(dims) if dim == 2]
    if len(xyzt_axes) != 1 or len(nu_axes) != 1:
        raise ValueError(
            f"Could not infer layout from shape {shape}; expected one axis with size 4 (xyzt) and one with size 2 (nu/nubar)."
        )

    axis_xyzt = xyzt_axes[0]
    axis_nu = nu_axes[0]
    remaining_axes = [i for i in range(3) if i not in {axis_xyzt, axis_nu}]
    if len(remaining_axes) != 1:
        raise ValueError(f"Could not infer flavor axis from shape {shape}")
    axis_flavor = remaining_axes[0]
    return LayoutSpec(axis_nu=axis_nu, axis_flavor=axis_flavor, axis_xyzt=axis_xyzt)


def raw_to_model_layout(raw: np.ndarray, layout: LayoutSpec) -> np.ndarray:
    # raw shape: [N, *, *, *], model shape: [N, nu/nubar, flavor, xyzt]
    return np.transpose(raw, (0, 1 + layout.axis_nu, 1 + layout.axis_flavor, 1 + layout.axis_xyzt))


def model_to_raw_layout(model: np.ndarray, layout: LayoutSpec) -> np.ndarray:
    # model shape: [N, nu/nubar, flavor, xyzt], raw shape: [N, *, *, *]
    axis_map = {
        layout.axis_nu: 1,
        layout.axis_flavor: 2,
        layout.axis_xyzt: 3,
    }
    return np.transpose(model, (0, axis_map[0], axis_map[1], axis_map[2]))


def normalize_model_layout(model_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ntot = np.sum(model_batch[:, :, :, 3], axis=(1, 2), keepdims=False).astype(np.float32)
    ntot = np.clip(ntot, 1.0e-12, None)
    norm_batch = (model_batch / ntot[:, None, None, None]).astype(np.float32)
    return norm_batch, ntot


def normalized_flat_features_from_raw(raw_batch: np.ndarray, layout: LayoutSpec) -> np.ndarray:
    model_batch = raw_to_model_layout(raw_batch, layout)
    norm_batch, _ = normalize_model_layout(model_batch)
    return norm_batch.reshape(norm_batch.shape[0], -1)


def derive_stability_labels(
    handle: h5py.File,
    n_samples: int,
    stability_threshold: float,
) -> np.ndarray:
    if STABLE_KEY in handle:
        stable = _as_float32(handle[STABLE_KEY][:n_samples]).reshape(-1)
        return stable

    if GROWTH_KEY in handle:
        growth = _as_float32(handle[GROWTH_KEY][:n_samples]).reshape(-1)
        stable = (growth <= stability_threshold).astype(np.float32)
        return stable

    # Fallback for files with no stable/growth annotations.
    return np.zeros((n_samples,), dtype=np.float32)


def _compression_arg(raw: str) -> str | None:
    value = raw.strip().lower()
    if value in {"none", "off", "no"}:
        return None
    return raw


def _n_samples_for_file(dataset: h5py.Dataset, max_samples_per_file: int | None) -> int:
    n_total = int(dataset.shape[0])
    if max_samples_per_file is None:
        return n_total
    return min(n_total, int(max_samples_per_file))


def _stable_mask(stable_labels: np.ndarray) -> np.ndarray:
    return stable_labels.reshape(-1) > 0.5


def select_files(
    input_dir: Path,
    explicit_files: list[str] | None,
    include_box3d: bool,
    include_leakagerates: bool,
) -> list[Path]:
    if explicit_files:
        candidates = [(input_dir / name).resolve() for name in explicit_files]
    else:
        candidates = sorted(input_dir.glob("*.h5"))

    selected: list[Path] = []
    for candidate in candidates:
        if not candidate.exists():
            raise FileNotFoundError(f"Input file not found: {candidate}")
        if (not include_box3d) and ("box3d" in candidate.name.lower()):
            print(f"[skip-box3d-name] {candidate.name}")
            continue
        if (not include_leakagerates) and ("leakagerates" in candidate.name.lower()):
            print(f"[skip-leakagerates-name] {candidate.name}")
            continue
        selected.append(candidate)
    return selected


def process_file(
    src_path: Path,
    dst_path: Path,
    batch_size: int,
    max_samples_per_file: int | None,
    stability_threshold: float,
    compression: str | None,
    bc_weight_override: np.ndarray | None = None,
    ambiguity_affects_vector: bool = True,
) -> None:
    with h5py.File(src_path, "r") as src:
        if F4_INITIAL_KEY not in src:
            print(f"[skip-missing-{F4_INITIAL_KEY}] {src_path.name}")
            return

        f4_in = src[F4_INITIAL_KEY]
        layout = infer_layout(tuple(f4_in.shape))
        n_samples = _n_samples_for_file(f4_in, max_samples_per_file)
        if n_samples < 1:
            print(f"[skip-empty] {src_path.name}")
            return

        stable = derive_stability_labels(src, n_samples=n_samples, stability_threshold=stability_threshold)
        if stable.shape[0] != n_samples:
            raise ValueError(f"Stable label length mismatch for {src_path.name}: {stable.shape[0]} vs {n_samples}")

        if bc_weight_override is not None:
            bc_weight_override = np.asarray(bc_weight_override, dtype=np.float32).reshape(-1)
            if bc_weight_override.shape[0] != n_samples:
                raise ValueError(
                    f"BC weight length mismatch for {src_path.name}: {bc_weight_override.shape[0]} vs {n_samples}"
                )

        if GROWTH_KEY in src:
            source_growth = _as_float32(src[GROWTH_KEY][:n_samples]).reshape(-1)
        else:
            source_growth = None
        has_true_final = F4_FINAL_KEY in src
        has_true_growth = source_growth is not None

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(dst_path, "w") as dst:
            dst.attrs["source_file"] = str(src_path.resolve())
            dst.attrs["n_samples"] = n_samples
            dst.attrs["layout_axis_nu"] = layout.axis_nu
            dst.attrs["layout_axis_flavor"] = layout.axis_flavor
            dst.attrs["layout_axis_xyzt"] = layout.axis_xyzt
            dst.attrs["targets_are_absolute"] = 1

            if "nf" in src:
                dst.create_dataset("nf", data=np.asarray(src["nf"]))
            elif has_true_final:
                dst.create_dataset("nf", data=np.asarray(src[F4_FINAL_KEY].shape[1 + layout.axis_flavor]))

            dst_f4_initial = dst.create_dataset(
                F4_INITIAL_KEY,
                shape=(n_samples, *f4_in.shape[1:]),
                dtype=np.float32,
                compression=compression,
            )
            dst_target_f4 = dst.create_dataset(
                TARGET_F4_KEY,
                shape=(n_samples, *f4_in.shape[1:]),
                dtype=np.float32,
                compression=compression,
            )
            dst_target_growth = dst.create_dataset(
                TARGET_GROWTH_KEY,
                shape=(n_samples,),
                dtype=np.float32,
                compression=compression,
            )
            dst_bc_weight = dst.create_dataset(
                MASK_BC_WEIGHT_KEY,
                shape=(n_samples,),
                dtype=np.float32,
                compression=compression,
            )
            dst_vector_weight = dst.create_dataset(
                MASK_VECTOR_WEIGHT_KEY,
                shape=(n_samples,),
                dtype=np.float32,
                compression=compression,
            )
            dst_reg_weight = dst.create_dataset(
                MASK_REG_WEIGHT_KEY,
                shape=(n_samples,),
                dtype=np.float32,
                compression=compression,
            )
            dst.create_dataset(
                STABLE_KEY,
                data=stable,
                dtype=np.float32,
                compression=compression,
            )

            dst_norm_f4_initial = dst.create_dataset(
                NORM_F4_INITIAL_KEY,
                shape=(n_samples, *f4_in.shape[1:]),
                dtype=np.float32,
                compression=compression,
            )
            dst_norm_target_f4 = dst.create_dataset(
                NORM_TARGET_F4_KEY,
                shape=(n_samples, *f4_in.shape[1:]),
                dtype=np.float32,
                compression=compression,
            )
            dst_norm_target_growth = dst.create_dataset(
                NORM_TARGET_GROWTH_KEY,
                shape=(n_samples,),
                dtype=np.float32,
                compression=compression,
            )

            if source_growth is not None:
                dst.create_dataset(
                    f"source/{GROWTH_KEY}",
                    data=source_growth,
                    dtype=np.float32,
                    compression=compression,
                )
            if has_true_final:
                dst.create_dataset(
                    f"source/{F4_FINAL_KEY}",
                    data=_as_float32(src[F4_FINAL_KEY][:n_samples]),
                    dtype=np.float32,
                    compression=compression,
                )

            for start in range(0, n_samples, batch_size):
                stop = min(start + batch_size, n_samples)
                raw_batch = _as_float32(f4_in[start:stop])
                model_batch = raw_to_model_layout(raw_batch, layout)

                if has_true_final:
                    target_raw_batch = _as_float32(src[F4_FINAL_KEY][start:stop])
                else:
                    # Stable-only files: enforce control-like identity target for flux.
                    target_raw_batch = raw_batch.copy()

                target_model_batch = raw_to_model_layout(target_raw_batch, layout)

                norm_batch_np, ntot = normalize_model_layout(model_batch)
                norm_target_batch_np = (target_model_batch / ntot[:, None, None, None]).astype(np.float32)

                target_raw = target_raw_batch.astype(np.float32, copy=False)
                if has_true_growth:
                    target_growth = source_growth[start:stop].astype(np.float32, copy=False)
                    reg_weight = np.ones((stop - start,), dtype=np.float32)
                else:
                    target_growth = np.zeros((stop - start,), dtype=np.float32)
                    reg_weight = np.ones((stop - start,), dtype=np.float32)

                if bc_weight_override is None:
                    bc_weight = np.ones((stop - start,), dtype=np.float32)
                else:
                    bc_weight = bc_weight_override[start:stop]

                if ambiguity_affects_vector:
                    vector_weight = bc_weight
                else:
                    vector_weight = np.ones((stop - start,), dtype=np.float32)

                norm_target_growth = (target_growth / ntot).astype(np.float32)

                norm_f4_raw = model_to_raw_layout(norm_batch_np, layout)
                norm_target_f4_raw = model_to_raw_layout(norm_target_batch_np, layout)

                dst_f4_initial[start:stop] = raw_batch
                dst_target_f4[start:stop] = target_raw
                dst_target_growth[start:stop] = target_growth
                dst_bc_weight[start:stop] = bc_weight
                dst_vector_weight[start:stop] = vector_weight
                dst_reg_weight[start:stop] = reg_weight
                dst_norm_f4_initial[start:stop] = norm_f4_raw
                dst_norm_target_f4[start:stop] = norm_target_f4_raw
                dst_norm_target_growth[start:stop] = norm_target_growth

    print(f"[processed] {src_path.name} -> {dst_path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare mixed stable/asymptotic HDF5 files with normalized targets and task masks."
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing source .h5 files")
    parser.add_argument("--output_dir", required=True, help="Directory for processed .h5 files")
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional list of filenames inside input_dir to process",
    )
    parser.add_argument(
        "--include_box3d_files",
        action="store_true",
        help="Process files with 'box3d' in the name (default is to skip them)",
    )
    parser.add_argument(
        "--include_leakagerates_files",
        action="store_true",
        help="Process files with 'leakagerates' in the name (default is to skip them)",
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="Chunk size for reading/writing")
    parser.add_argument(
        "--max_samples_per_file",
        type=int,
        default=None,
        help="Optional cap for processed samples per file (for smoke testing)",
    )
    parser.add_argument(
        "--stability_threshold",
        type=float,
        default=0.0,
        help="When source stable labels are missing, derive stable=(source growth<=threshold)",
    )
    parser.add_argument(
        "--compression",
        default="lzf",
        help="HDF5 compression (e.g. lzf, gzip, none)",
    )
    parser.add_argument(
        "--disable_ambiguity_filter",
        action="store_true",
        help="Disable stable-vs-unstable nearest-neighbor ambiguity downweighting.",
    )
    parser.add_argument(
        "--ambiguity_quantile",
        type=float,
        default=0.02,
        help="Stable d_min quantile used as ambiguity threshold when absolute threshold is not provided.",
    )
    parser.add_argument(
        "--ambiguity_distance_threshold",
        type=float,
        default=None,
        help="Absolute L2 threshold on normalized flattened F4 features for ambiguous stable points.",
    )
    parser.add_argument(
        "--ambiguity_stable_weight",
        type=float,
        default=0.0,
        help="Weight assigned to ambiguous stable points (for BC, and optionally vector).",
    )
    parser.add_argument(
        "--ambiguity_only_bc",
        action="store_true",
        help="Apply ambiguity downweighting only to BC task; keep vector target weight at 1.",
    )
    parser.add_argument(
        "--ambiguity_max_unstable_points",
        type=int,
        default=300000,
        help="Cap on unstable reference points used for nearest-neighbor matching (0 disables cap).",
    )
    parser.add_argument(
        "--ambiguity_random_seed",
        type=int,
        default=42,
        help="Random seed used when sampling unstable reference points.",
    )
    parser.add_argument(
        "--ambiguity_bruteforce_chunk_size",
        type=int,
        default=4096,
        help="Reference chunk size for brute-force nearest-neighbor fallback.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if not 0.0 <= float(args.ambiguity_quantile) <= 1.0:
        raise ValueError("--ambiguity_quantile must be in [0, 1]")
    if float(args.ambiguity_stable_weight) < 0.0:
        raise ValueError("--ambiguity_stable_weight must be >= 0")
    if args.ambiguity_bruteforce_chunk_size < 1:
        raise ValueError("--ambiguity_bruteforce_chunk_size must be >= 1")
    if args.ambiguity_max_unstable_points < 0:
        raise ValueError("--ambiguity_max_unstable_points must be >= 0")

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = select_files(
        input_dir=input_dir,
        explicit_files=args.files,
        include_box3d=bool(args.include_box3d_files),
        include_leakagerates=bool(args.include_leakagerates_files),
    )
    if not files:
        print("No files selected for processing.")
        return

    ambiguity_weights_by_file: dict[str, np.ndarray] = {}
    if not bool(args.disable_ambiguity_filter):
        ambiguity_weights_by_file = prepare_ambiguity_weights(
            files=files,
            f4_initial_key=F4_INITIAL_KEY,
            batch_size=int(args.batch_size),
            max_samples_per_file=args.max_samples_per_file,
            stability_threshold=float(args.stability_threshold),
            ambiguity_quantile=float(args.ambiguity_quantile),
            ambiguity_distance_threshold=args.ambiguity_distance_threshold,
            ambiguity_stable_weight=float(args.ambiguity_stable_weight),
            ambiguity_max_unstable_points=int(args.ambiguity_max_unstable_points),
            ambiguity_random_seed=int(args.ambiguity_random_seed),
            ambiguity_bruteforce_chunk_size=int(args.ambiguity_bruteforce_chunk_size),
            as_float32=_as_float32,
            infer_layout=infer_layout,
            n_samples_for_file=_n_samples_for_file,
            derive_stability_labels=derive_stability_labels,
            stable_mask=_stable_mask,
            normalized_flat_features_from_raw=normalized_flat_features_from_raw,
        )
    else:
        print("[ambiguity] disabled by --disable_ambiguity_filter")

    compression = _compression_arg(args.compression)
    for src_path in files:
        dst_path = output_dir / src_path.name
        if dst_path.exists() and not args.overwrite:
            print(f"[skip-existing] {dst_path}")
            continue
        process_file(
            src_path=src_path,
            dst_path=dst_path,
            batch_size=int(args.batch_size),
            max_samples_per_file=args.max_samples_per_file,
            stability_threshold=float(args.stability_threshold),
            compression=compression,
            bc_weight_override=ambiguity_weights_by_file.get(str(src_path.resolve())),
            ambiguity_affects_vector=not bool(args.ambiguity_only_bc),
        )


if __name__ == "__main__":
    main()
