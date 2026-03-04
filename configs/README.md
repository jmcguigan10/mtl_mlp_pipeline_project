# Config Guide

## Start Here (Rhea MLP, absolute targets)

1. Preprocess raw datasets once:

```bash
make preprocess INPUT_DIR=./example_data OUTPUT_DIR=./example_data_box3d_abs
```

2. Edit file-level splits in `data.train_files`, `data.val_files`, `data.test_files` inside:

`configs/rhea_box3d_abs_train.yaml`

3. Start training:

```bash
make train CONFIG=configs/rhea_box3d_abs_train.yaml
```

## Which Config To Use

- `rhea_box3d_abs_train.yaml`: main non-equivariant MLP training config (recommended default).
- `rhea_equivariant_abs_train.yaml`: non-GNN equivariant-basis training config.
- `rhea_v100_sweep.yaml`: leakage-aware split used for V100 tuning sweeps.
- `rhea_v100_recommended.yaml`: recommended V100 starting config from sweep results.
- `rhea_v100_recommended_full_template.yaml`: same as recommended, but pointing at `../full_data_box3d_abs/*`.
- `rhea_box3d_abs_smoke.yaml`: quick one-epoch pipeline smoke check.
- `rhea_equivariant_abs_smoke.yaml`: quick one-epoch equivariant smoke check.
- `rhea_stable_smoke.yaml`: stable-only identity-target smoke check.
- `example_*.yaml`: synthetic/tutorial configs using dummy data (`inputs`, `targets/*` layout).

## Minimum Edits Before Real Training

- `output.experiment_name`: set a unique run name.
- `data.train_files`, `data.val_files`, `data.test_files`: define file-level splits to prevent leakage.
- `training.epochs`: set your full run budget.
- `data.loader.batch_size`: tune for your GPU memory.

## Key Fields

- `data.keys.*`: maps HDF5 paths to model inputs/targets.
- `data.preprocess.reshape`: flatten controls for tensor-shaped datasets.
- `multitask.loss_balancer.name`: `static`, `kendall_gal`, or `gradnorm`.
- `multitask.gradient_surgery.name`: `none` or `pcgrad`.
- `evaluation.control.enabled`: computes Box3D control metrics during evaluation.

## V100 Sweep Command

```bash
make tune-v100 SWEEP_CONFIG=configs/rhea_v100_sweep.yaml
```
