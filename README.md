# Multi-task MLP pipeline

A configurable PyTorch project for a **24-dimensional input** with three heads:

- **BC head** (default example assumes binary classification)
- **Vector regression head**
- **Scalar regression head**

Everything is plain MLPs. No transformers sneaking in to feel important.

## Included features

- shared MLP trunk plus three task heads
- YAML-based configuration
- custom **multi-file HDF5 dataset** with lazy file opening
- CUDA-aware device handling
- optional CPU thread controls for saner local testing
- optional mixed precision for standard/Kendall-Gal training
- per-section controls for:
  - hidden sizes
  - activation
  - batch norm
  - layer norm
  - dropout
  - residual blocks
  - freeze flags
  - feature **recalibration** blocks
- optimizer parameter-group overrides per module path
- AdamW / Adam / SGD optimizer factory
- cosine / step / plateau / one-cycle schedulers
- checkpointing, CSV history, JSON metrics
- static loss weights, **Kendall-Gal**, **GradNorm**, and **PCGrad** options
- train / evaluate / predict scripts
- synthetic HDF5 generator for smoke tests

## Assumptions worth stating before somebody blames the repo

- I interpreted **"bc"** as a classification head in the example configs. The head itself is just an MLP logits head, so you can switch the BC loss to `cross_entropy` and change `output_dim` if you want multiclass instead of binary.
- I interpreted **"recal"** as an optional **feature recalibration block** for MLP features. It behaves like a small gating module that rescales activations.
- In this starter implementation, **PCGrad is wired to the static loss balancer**. That keeps the optimizer behavior clean instead of turning the training loop into ceremonial nonsense.
- Mixed precision is **auto-disabled** for PCGrad and GradNorm in this starter version because manual gradient surgery / gradient-norm control and AMP together are more brittle than most human group projects.

## Project layout

```text
mtl_mlp_pipeline_project/
├── Makefile
├── configs/
│   ├── README.md
│   ├── rhea_box3d_abs_train.yaml
│   ├── rhea_equivariant_abs_train.yaml
│   ├── rhea_box3d_abs_smoke.yaml
│   ├── rhea_equivariant_abs_smoke.yaml
│   ├── rhea_stable_smoke.yaml
│   └── example_*.yaml
├── mtl_mlp/
│   ├── config.py
│   ├── data/
│   │   └── hdf5_dataset.py
│   ├── models/
│   │   ├── blocks.py
│   │   ├── heads.py
│   │   └── multitask_model.py
│   ├── preprocessing/
│   │   ├── ambiguity_filter.py
│   │   ├── box3d_heuristic.py
│   │   └── lebedev17_fallback.py
│   ├── training/
│   │   ├── balancers.py
│   │   ├── epoch_metrics.py
│   │   ├── losses.py
│   │   ├── optim.py
│   │   ├── pcgrad.py
│   │   └── trainer.py
│   └── utils/
│       └── common.py
├── scripts/
│   ├── make_dummy_hdf5.py
│   └── smoke_test.py
├── train.py
├── evaluate.py
├── predict.py
└── requirements.txt
```

## HDF5 format

Each file can contain datasets at configurable paths. The example configs assume:

```text
/inputs              float32 [N, 24]
/targets/bc          float32 [N, 1]      # binary classification labels in the example
/targets/vector      float32 [N, D]
/targets/reg         float32 [N, 1]
```

Optional sample weights are also supported if you add a dataset path in the YAML:

```yaml
data:
  keys:
    sample_weight: weights
```

## Quick start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) If you are training on Rhea-style data (recommended path)

```bash
make preprocess INPUT_DIR=./example_data OUTPUT_DIR=./example_data_box3d_abs
make train CONFIG=configs/rhea_box3d_abs_train.yaml
```

Config guide:
- `configs/README.md`

### 3) Synthetic demo path (optional)

```bash
make check-python
./.venv/bin/python scripts/make_dummy_hdf5.py --output_dir ./example_data --vector_dim 3
```

### 4) Train

```bash
make train CONFIG=configs/example_static.yaml
```

### 5) Evaluate a checkpoint

```bash
make eval \
  CONFIG=configs/example_static.yaml \
  CHECKPOINT=outputs/static_demo/checkpoints/best.pt \
  SPLIT=test \
  OUTPUT=outputs/static_demo/test_metrics.json
```

### 6) Run inference and save predictions

```bash
make predict \
  CONFIG=configs/example_static.yaml \
  CHECKPOINT=outputs/static_demo/checkpoints/best.pt \
  SPLIT=test \
  OUTPUT=outputs/static_demo/test_predictions.npz
```

You can also pass `FILES="file1.h5 file2.h5"` to `make predict` for ad-hoc HDF5 inference. In that mode only the input dataset is required.

Dataset split control stays YAML-driven (to prevent leakage by file):
- `data.train_files`
- `data.val_files`
- `data.test_files`

`make train/eval/predict` only choose the config and runtime options; they do not override those split lists unless you provide a different YAML.

## Preprocessing for mixed stable/asymptotic training

To build MLP-ready datasets from Rhea-style files (`F4_initial(1|ccm)`), run:

```bash
make preprocess INPUT_DIR=./example_data OUTPUT_DIR=./example_data_box3d_abs
```

Behavior:
- preserves `F4_initial(1|ccm)` and `nf` in the output files
- writes absolute training targets under `targets/...`:
  - `targets/F4_final(1|ccm)` (from source asymptotic data; falls back to `F4_initial` for stable files)
  - `targets/growthRate(1|s)` (from source asymptotic data; falls back to `0` for stable files)
- writes per-task loss masks under `masks/...`:
  - `masks/bc_target_weight` (`1` by default, downweighted for stable points too close to unstable points)
  - `masks/vector_target_weight` (matches `bc_target_weight` by default; use `--ambiguity_only_bc` to keep this at `1`)
  - `masks/reg_target_weight` (`1` for asymptotic samples, `0` for stable samples)
- also writes normalized absolute columns under `normalized/...`:
  - `normalized/F4_initial(1|ccm)`
  - `normalized/targets/F4_final(1|ccm)`
  - `normalized/targets/growthRate(1|s)`
- writes/derives `stable` labels (uses source `stable` when present; otherwise derives from source growth threshold)
- skips any file with `box3d` in its filename unless `--include_box3d_files` is passed
- skips any file with `leakagerates` in its filename unless `--include_leakagerates_files` is passed
- runs a stable-vs-unstable nearest-neighbor ambiguity filter by default:
  - builds normalized flattened `F4_initial` features
  - computes each stable sample's nearest unstable distance
  - downweights ambiguous stable points using `masks/bc_target_weight` (and `masks/vector_target_weight` unless `--ambiguity_only_bc`)
  - defaults: quantile threshold `--ambiguity_quantile 0.02`, downweight `--ambiguity_stable_weight 0.0`

Use:
- `configs/rhea_box3d_abs_train.yaml` for full non-equivariant MLP training on absolute targets.
- `configs/rhea_equivariant_abs_train.yaml` for full non-GNN equivariant-basis training.
- `configs/rhea_box3d_abs_smoke.yaml` and `configs/rhea_equivariant_abs_smoke.yaml` for 1-epoch smoke checks.
- `configs/README.md` for profile descriptions and what to edit first.

When `evaluation.control.enabled: true`, Box3D control is computed on-the-fly during evaluation (no `box3d/*` dataset columns needed). Reported metrics include:
- `*/vector_vs_control_frac_mean|median|p95|p99`
- `*/growth_vs_control_frac_mean|median|p95|p99`
- and percentage forms `*/vector_error_vs_control_pct`, `*/growth_error_vs_control_pct`.

## Config notes

The main knobs live in these sections:

- `model.trunk`
- `model.heads.bc`
- `model.heads.vector_regression`
- `model.heads.regression`
- `multitask.loss_balancer`
- `multitask.gradient_surgery`
- `training.optimizer.param_groups`

Example per-module optimizer override:

```yaml
training:
  optimizer:
    name: adamw
    lr: 0.001
    weight_decay: 0.01
    param_groups:
      - module: trunk
        lr: 0.001
        weight_decay: 0.01
      - module: heads.bc
        lr: 0.0007
        weight_decay: 0.0
```

## Outputs

Training writes to:

```text
outputs/<experiment_name>/
├── checkpoints/
│   ├── best.pt
│   ├── last.pt
│   └── epoch_XXX.pt
├── config_snapshot.json
├── final_metrics.json
└── history.csv
```

## SLURM launchers (Rhea-compatible)

This repo includes Rhea-style sbatch wrappers with the same cluster defaults (`isaac-utk0307`, `condo-slagergr`, module stack, repo-local `.venv`). The wrappers now route execution through `make` targets:

```bash
sbatch train.sbatch
sbatch eval_f1.sbatch
sbatch predict.sbatch
sbatch preprocess_box3d.sbatch
```

Common overrides are passed at submit time, for example:

```bash
sbatch --export=ALL,MTL_MLP_CONFIG=configs/rhea_box3d_abs_train.yaml train.sbatch
sbatch --export=ALL,MTL_MLP_CONFIG=configs/rhea_box3d_abs_train.yaml,MTL_MLP_CHECKPOINT=outputs/rhea_box3d_abs_train/checkpoints/best.pt eval_f1.sbatch
sbatch --export=ALL,MTL_MLP_CHECKPOINT=outputs/rhea_box3d_abs_train/checkpoints/best.pt,MTL_MLP_FILES="example_data_box3d_abs/stable_random.h5" predict.sbatch
sbatch --export=ALL,MTL_MLP_INPUT_DIR=example_data,MTL_MLP_OUTPUT_DIR=example_data_box3d_abs preprocess_box3d.sbatch
```

## Smoke test

```bash
make smoke
```

That generates synthetic data in a temporary directory and runs a short training cycle. It does not modify `example_data/`.

For the Rhea dataset symlink layout (`example_data/*.h5 -> ../Rhea/datasets/*.h5`), use:

```bash
make smoke-rhea
```

This runs a one-epoch train/eval/predict cycle with `configs/rhea_stable_smoke.yaml`.

For Box3D preprocessing + absolute-target MLP training smoke test, use:

```bash
make smoke-box3d
```

For non-GNN equivariant-basis training + control metrics smoke test, use:

```bash
make smoke-equiv
make test-equiv
```
