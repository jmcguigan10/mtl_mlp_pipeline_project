# Large-Batch Trial Status (2026-03-04 15:54 EST)

## Scope
- Training configs:
  - `configs/rhea_v100_recommended_bs2048.yaml`
  - `configs/rhea_v100_recommended_bs4096.yaml`
  - `configs/rhea_v100_recommended_bs8192.yaml`
- Jobs launched via `train.sbatch`.

## Job Status Snapshot
- `5025670` (`bs2048`): `RUNNING` on `clrv1207` (`TIME=00:14:33`)
- `5025671` (`bs4096`): `RUNNING` on `clrv1207` (`TIME=00:14:33`)
- `5025672` (`bs8192`): `PENDING` (`Resources`)

## Latest Progress From Logs
- `5025670` (`bs2048`):
  - Latest parsed train progress: `train epoch 2`, `77/523` (7.7%), `total_loss=0.6088`
  - Latest parsed val progress: `val epoch 1`, `222/224` (99.1%), `total_loss=0.2091`
- `5025671` (`bs4096`):
  - Latest parsed train progress: `train epoch 1`, `261/262` (99.6%), `total_loss=0.7924`
  - Latest parsed val progress: `val epoch 1`, `111/112` (99.1%), `total_loss=0.2501`

## Epoch 1 Metrics (from `history.csv`)
- `bs2048` (`outputs/rhea_v100_recommended_bs2048/history.csv`):
  - `train/total_loss=0.6633375263705278`
  - `val/total_loss=0.6796947632363658`
  - `val/bc_f1=0.5682059554450114`
  - `val/vector_regression=0.0009356684617217282`
  - `val/regression=0.00031885863454555113`
- `bs4096` (`outputs/rhea_v100_recommended_bs4096/history.csv`):
  - `train/total_loss=0.6615413386449089`
  - `val/total_loss=0.6593523200335827`
  - `val/bc_f1=0.563068071296211`
  - `val/vector_regression=0.0011462545503176616`
  - `val/regression=0.00112257853939651`

## Checkpoint Artifacts
- `outputs/rhea_v100_recommended_bs2048/checkpoints/` contains `best.pt`, `epoch_001.pt`, `last.pt`.
- `outputs/rhea_v100_recommended_bs4096/checkpoints/` contains `best.pt`, `epoch_001.pt`, `last.pt`.

## Notes
- No `OOM`, traceback, or exception signatures were found in current `slurm/output/train-5025670.err` and `slurm/output/train-5025671.err`.
- `bs2048` is already into epoch 2 while `bs4096` is finishing epoch 1 validation.
- `bs8192` has not started yet due queue resources.
