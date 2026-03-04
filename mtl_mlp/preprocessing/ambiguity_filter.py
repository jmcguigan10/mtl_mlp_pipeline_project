from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

import h5py
import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


AsFloat32Fn = Callable[[Union[np.ndarray, h5py.Dataset]], np.ndarray]
InferLayoutFn = Callable[[tuple[int, ...]], Any]
SampleCountFn = Callable[[h5py.Dataset, Optional[int]], int]
DeriveStabilityFn = Callable[[h5py.File, int, float], np.ndarray]
StableMaskFn = Callable[[np.ndarray], np.ndarray]
NormalizedFeaturesFn = Callable[[np.ndarray, Any], np.ndarray]


@dataclass(frozen=True)
class NeighborIndex:
    points: np.ndarray
    backend: str
    tree: Any | None = None
    points64: np.ndarray | None = None
    points_norm2: np.ndarray | None = None


def build_neighbor_index(reference_points: np.ndarray) -> NeighborIndex:
    if reference_points.ndim != 2 or reference_points.shape[1] != 24:
        raise ValueError(f"Expected reference points shape [N,24], got {reference_points.shape}")

    if cKDTree is not None:
        tree = cKDTree(reference_points)
        return NeighborIndex(points=reference_points, backend="ckdtree", tree=tree)

    points64 = reference_points.astype(np.float64, copy=False)
    points_norm2 = np.sum(points64 * points64, axis=1)
    return NeighborIndex(
        points=reference_points,
        backend="bruteforce",
        tree=None,
        points64=points64,
        points_norm2=points_norm2,
    )


def query_min_distances(
    index: NeighborIndex,
    query_points: np.ndarray,
    brute_force_chunk_size: int,
) -> np.ndarray:
    if query_points.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    if query_points.ndim != 2 or query_points.shape[1] != 24:
        raise ValueError(f"Expected query points shape [N,24], got {query_points.shape}")

    if index.tree is not None:
        distances, _ = index.tree.query(query_points, k=1)
        return np.asarray(distances, dtype=np.float32).reshape(-1)

    if index.points64 is None or index.points_norm2 is None:
        raise ValueError("Bruteforce nearest-neighbor index is missing cached arrays.")
    if brute_force_chunk_size < 1:
        raise ValueError("--ambiguity_bruteforce_chunk_size must be >= 1")

    query64 = query_points.astype(np.float64, copy=False)
    query_norm2 = np.sum(query64 * query64, axis=1)
    min_dist_sq = np.full((query64.shape[0],), np.inf, dtype=np.float64)

    for start in range(0, index.points64.shape[0], brute_force_chunk_size):
        stop = min(start + brute_force_chunk_size, index.points64.shape[0])
        ref_chunk = index.points64[start:stop]
        ref_norm2_chunk = index.points_norm2[start:stop]
        dist_sq = query_norm2[:, None] + ref_norm2_chunk[None, :] - 2.0 * (query64 @ ref_chunk.T)
        np.maximum(dist_sq, 0.0, out=dist_sq)
        min_dist_sq = np.minimum(min_dist_sq, np.min(dist_sq, axis=1))

    np.sqrt(min_dist_sq, out=min_dist_sq)
    return min_dist_sq.astype(np.float32)


def collect_unstable_reference_points(
    files: list[Path],
    *,
    f4_initial_key: str,
    batch_size: int,
    max_samples_per_file: int | None,
    stability_threshold: float,
    max_unstable_points: int,
    random_seed: int,
    as_float32: AsFloat32Fn,
    infer_layout: InferLayoutFn,
    n_samples_for_file: SampleCountFn,
    derive_stability_labels: DeriveStabilityFn,
    stable_mask: StableMaskFn,
    normalized_flat_features_from_raw: NormalizedFeaturesFn,
) -> np.ndarray:
    unstable_chunks: list[np.ndarray] = []

    for src_path in files:
        with h5py.File(src_path, "r") as src:
            if f4_initial_key not in src:
                continue

            f4_in = src[f4_initial_key]
            layout = infer_layout(tuple(f4_in.shape))
            n_samples = n_samples_for_file(f4_in, max_samples_per_file)
            if n_samples < 1:
                continue

            stable = derive_stability_labels(src, n_samples=n_samples, stability_threshold=stability_threshold)
            stable_mask_all = stable_mask(stable)
            if not np.any(~stable_mask_all):
                continue

            for start in range(0, n_samples, batch_size):
                stop = min(start + batch_size, n_samples)
                stable_batch_mask = stable_mask_all[start:stop]
                unstable_batch_mask = ~stable_batch_mask
                if not np.any(unstable_batch_mask):
                    continue

                raw_batch = as_float32(f4_in[start:stop])[unstable_batch_mask]
                unstable_chunks.append(normalized_flat_features_from_raw(raw_batch, layout))

    if not unstable_chunks:
        return np.empty((0, 24), dtype=np.float32)

    unstable = np.concatenate(unstable_chunks, axis=0).astype(np.float32, copy=False)

    if max_unstable_points > 0 and unstable.shape[0] > max_unstable_points:
        rng = np.random.default_rng(int(random_seed))
        keep_idx = rng.choice(unstable.shape[0], size=max_unstable_points, replace=False)
        unstable = unstable[keep_idx]
        print(
            f"[ambiguity] sampled unstable reference points: kept {unstable.shape[0]} "
            f"of {sum(chunk.shape[0] for chunk in unstable_chunks)}"
        )
    else:
        print(f"[ambiguity] unstable reference points: {unstable.shape[0]}")

    return unstable


def collect_stable_neighbor_distances(
    files: list[Path],
    *,
    f4_initial_key: str,
    index: NeighborIndex,
    batch_size: int,
    max_samples_per_file: int | None,
    stability_threshold: float,
    brute_force_chunk_size: int,
    as_float32: AsFloat32Fn,
    infer_layout: InferLayoutFn,
    n_samples_for_file: SampleCountFn,
    derive_stability_labels: DeriveStabilityFn,
    stable_mask: StableMaskFn,
    normalized_flat_features_from_raw: NormalizedFeaturesFn,
) -> np.ndarray:
    stable_distance_chunks: list[np.ndarray] = []

    for src_path in files:
        with h5py.File(src_path, "r") as src:
            if f4_initial_key not in src:
                continue

            f4_in = src[f4_initial_key]
            layout = infer_layout(tuple(f4_in.shape))
            n_samples = n_samples_for_file(f4_in, max_samples_per_file)
            if n_samples < 1:
                continue

            stable = derive_stability_labels(src, n_samples=n_samples, stability_threshold=stability_threshold)
            stable_mask_all = stable_mask(stable)
            if not np.any(stable_mask_all):
                continue

            for start in range(0, n_samples, batch_size):
                stop = min(start + batch_size, n_samples)
                stable_batch_mask = stable_mask_all[start:stop]
                if not np.any(stable_batch_mask):
                    continue

                raw_batch = as_float32(f4_in[start:stop])[stable_batch_mask]
                stable_features = normalized_flat_features_from_raw(raw_batch, layout)
                stable_distance_chunks.append(
                    query_min_distances(
                        index=index,
                        query_points=stable_features,
                        brute_force_chunk_size=brute_force_chunk_size,
                    )
                )

    if not stable_distance_chunks:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(stable_distance_chunks, axis=0).astype(np.float32, copy=False)


def build_ambiguity_weights_by_file(
    files: list[Path],
    *,
    f4_initial_key: str,
    index: NeighborIndex,
    threshold: float,
    stable_weight: float,
    batch_size: int,
    max_samples_per_file: int | None,
    stability_threshold: float,
    brute_force_chunk_size: int,
    as_float32: AsFloat32Fn,
    infer_layout: InferLayoutFn,
    n_samples_for_file: SampleCountFn,
    derive_stability_labels: DeriveStabilityFn,
    stable_mask: StableMaskFn,
    normalized_flat_features_from_raw: NormalizedFeaturesFn,
) -> dict[str, np.ndarray]:
    weights_by_file: dict[str, np.ndarray] = {}
    total_stable = 0
    total_ambiguous = 0

    for src_path in files:
        with h5py.File(src_path, "r") as src:
            if f4_initial_key not in src:
                continue

            f4_in = src[f4_initial_key]
            layout = infer_layout(tuple(f4_in.shape))
            n_samples = n_samples_for_file(f4_in, max_samples_per_file)
            if n_samples < 1:
                continue

            stable = derive_stability_labels(src, n_samples=n_samples, stability_threshold=stability_threshold)
            stable_mask_all = stable_mask(stable)
            n_stable = int(np.count_nonzero(stable_mask_all))
            file_weights = np.ones((n_samples,), dtype=np.float32)
            n_ambiguous = 0

            if n_stable > 0:
                for start in range(0, n_samples, batch_size):
                    stop = min(start + batch_size, n_samples)
                    stable_batch_mask = stable_mask_all[start:stop]
                    if not np.any(stable_batch_mask):
                        continue

                    local_idx = np.flatnonzero(stable_batch_mask)
                    raw_batch = as_float32(f4_in[start:stop])[stable_batch_mask]
                    stable_features = normalized_flat_features_from_raw(raw_batch, layout)
                    distances = query_min_distances(
                        index=index,
                        query_points=stable_features,
                        brute_force_chunk_size=brute_force_chunk_size,
                    )
                    ambiguous_mask = distances <= threshold
                    if np.any(ambiguous_mask):
                        ambiguous_idx = start + local_idx[ambiguous_mask]
                        file_weights[ambiguous_idx] = stable_weight
                        n_ambiguous += int(np.count_nonzero(ambiguous_mask))

            total_stable += n_stable
            total_ambiguous += n_ambiguous
            weights_by_file[str(src_path.resolve())] = file_weights

            frac = float(n_ambiguous) / max(n_stable, 1)
            print(
                f"[ambiguity-file] {src_path.name}: stable={n_stable}, "
                f"ambiguous={n_ambiguous}, frac={frac:.4f}"
            )

    total_frac = float(total_ambiguous) / max(total_stable, 1)
    print(
        f"[ambiguity-summary] stable={total_stable}, ambiguous={total_ambiguous}, "
        f"frac={total_frac:.4f}, stable_weight={stable_weight:.3f}, threshold={threshold:.6e}"
    )
    return weights_by_file


def prepare_ambiguity_weights(
    files: list[Path],
    *,
    f4_initial_key: str,
    batch_size: int,
    max_samples_per_file: int | None,
    stability_threshold: float,
    ambiguity_quantile: float,
    ambiguity_distance_threshold: float | None,
    ambiguity_stable_weight: float,
    ambiguity_max_unstable_points: int,
    ambiguity_random_seed: int,
    ambiguity_bruteforce_chunk_size: int,
    as_float32: AsFloat32Fn,
    infer_layout: InferLayoutFn,
    n_samples_for_file: SampleCountFn,
    derive_stability_labels: DeriveStabilityFn,
    stable_mask: StableMaskFn,
    normalized_flat_features_from_raw: NormalizedFeaturesFn,
) -> dict[str, np.ndarray]:
    unstable_points = collect_unstable_reference_points(
        files=files,
        f4_initial_key=f4_initial_key,
        batch_size=batch_size,
        max_samples_per_file=max_samples_per_file,
        stability_threshold=stability_threshold,
        max_unstable_points=ambiguity_max_unstable_points,
        random_seed=ambiguity_random_seed,
        as_float32=as_float32,
        infer_layout=infer_layout,
        n_samples_for_file=n_samples_for_file,
        derive_stability_labels=derive_stability_labels,
        stable_mask=stable_mask,
        normalized_flat_features_from_raw=normalized_flat_features_from_raw,
    )
    if unstable_points.shape[0] == 0:
        print("[ambiguity] no unstable points found, leaving bc/vector weights at 1.")
        return {}

    index = build_neighbor_index(unstable_points)
    print(f"[ambiguity] nearest-neighbor backend: {index.backend}")

    stable_distances = collect_stable_neighbor_distances(
        files=files,
        f4_initial_key=f4_initial_key,
        index=index,
        batch_size=batch_size,
        max_samples_per_file=max_samples_per_file,
        stability_threshold=stability_threshold,
        brute_force_chunk_size=ambiguity_bruteforce_chunk_size,
        as_float32=as_float32,
        infer_layout=infer_layout,
        n_samples_for_file=n_samples_for_file,
        derive_stability_labels=derive_stability_labels,
        stable_mask=stable_mask,
        normalized_flat_features_from_raw=normalized_flat_features_from_raw,
    )
    if stable_distances.shape[0] == 0:
        print("[ambiguity] no stable points found, leaving bc/vector weights at 1.")
        return {}

    if ambiguity_distance_threshold is not None:
        threshold = float(ambiguity_distance_threshold)
        threshold_mode = "absolute"
    else:
        threshold = float(np.quantile(stable_distances, float(ambiguity_quantile)))
        threshold_mode = f"quantile(q={float(ambiguity_quantile):.4f})"

    print(
        f"[ambiguity-threshold] mode={threshold_mode}, threshold={threshold:.6e}, "
        f"stable_dmin(min/median/max)="
        f"{float(np.min(stable_distances)):.6e}/"
        f"{float(np.median(stable_distances)):.6e}/"
        f"{float(np.max(stable_distances)):.6e}"
    )

    return build_ambiguity_weights_by_file(
        files=files,
        f4_initial_key=f4_initial_key,
        index=index,
        threshold=threshold,
        stable_weight=float(ambiguity_stable_weight),
        batch_size=batch_size,
        max_samples_per_file=max_samples_per_file,
        stability_threshold=stability_threshold,
        brute_force_chunk_size=ambiguity_bruteforce_chunk_size,
        as_float32=as_float32,
        infer_layout=infer_layout,
        n_samples_for_file=n_samples_for_file,
        derive_stability_labels=derive_stability_labels,
        stable_mask=stable_mask,
        normalized_flat_features_from_raw=normalized_flat_features_from_raw,
    )
