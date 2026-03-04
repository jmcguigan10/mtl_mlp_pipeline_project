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
├── configs/
│   ├── example_static.yaml
│   ├── example_kendall_gal.yaml
│   ├── example_gradnorm.yaml
│   └── example_pcgrad.yaml
├── mtl_mlp/
│   ├── config.py
│   ├── data/
│   │   └── hdf5_dataset.py
│   ├── models/
│   │   ├── blocks.py
│   │   ├── heads.py
│   │   └── multitask_model.py
│   ├── training/
│   │   ├── balancers.py
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

### 2) Make synthetic HDF5 data

```bash
python scripts/make_dummy_hdf5.py --output_dir ./example_data --vector_dim 3
```

### 3) Train

```bash
python train.py --config configs/example_static.yaml
```

### 4) Evaluate a checkpoint

```bash
python evaluate.py --config configs/example_static.yaml --checkpoint outputs/static_demo/checkpoints/best.pt --split test
```

### 5) Run inference and save predictions

```bash
python predict.py --config configs/example_static.yaml --checkpoint outputs/static_demo/checkpoints/best.pt --split test --output outputs/static_demo/test_predictions.npz
```

You can also pass `--files file1.h5 file2.h5` to run prediction on arbitrary HDF5 files. In that mode only the input dataset is required, because sometimes you just want predictions instead of another lecture from your labels.

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

## Smoke test

```bash
python scripts/smoke_test.py
```

That generates synthetic data and runs a short training cycle using `configs/example_static.yaml`. Miracles do occasionally happen.
