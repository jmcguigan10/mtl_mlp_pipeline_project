# mtl_mlp_pipeline_project review and recommended patch set

## Bottom line

I would **not** spend the next round changing the core MLP architecture.
The current repo is already clean enough structurally. The bigger problems are in:

1. **evaluation correctness**
2. **metric correctness under masks / zero baselines**
3. **batch-size tuning logic**
4. **HDF5 input throughput for huge, unchunked files**

The preliminary evidence in the repo points pretty clearly in that direction:

- `outputs/tuning/v100_sweep_results.csv` shows **throughput is basically flat across batch sizes 256 to 8192** while train loss gets worse as batch size increases. So the current choice of `1024` is not actually justified by speed.
- `scripts/v100_tune_sweep.py` currently runs the batch sweep with **`val_steps=0`** and then selects the batch size **purely by throughput**.
- `outputs/rhea_box3d_abs_train/final_metrics.json` says **`best_epoch = 2`** but **`epoch = 10`**, which strongly suggests the final test evaluation is being done on the **last** weights, not the **best** weights.
- `configs/rhea_v100_recommended.yaml` uses `masks/bc_target_weight`, `masks/vector_target_weight`, and `masks/reg_target_weight`, but `mtl_mlp/training/epoch_metrics.py` only applies weights to regression metrics. **BC metrics currently ignore the BC mask.**
- The same metrics file generates absurd control ratios when the control baseline error is effectively zero. In `outputs/rhea_box3d_abs_train/final_metrics.json`, `test/control_reg_mae` is `0.0` but `test/reg_vs_control_frac_mean` is enormous. That metric is not useful in its current form.

## Priority order

1. **Fix evaluation correctness first**
2. **Fix masked metrics**
3. **Fix the sweep logic**
4. **Fix HDF5 sampling / loading pattern**
5. **Only then retune configs**

---

## Patch 1: restore the best checkpoint before final test evaluation

### Why

Right now the trainer saves `best.pt`, but `fit()` appears to run test evaluation on the current in-memory model at the end of training. With early stopping, that means you can easily report test metrics from an overfit tail epoch instead of the best epoch.

### Files

- `mtl_mlp/training/trainer.py`

### Diff

```diff
--- a/mtl_mlp/training/trainer.py
+++ b/mtl_mlp/training/trainer.py
@@
-from ..utils.common import count_parameters, ensure_dir, get_device, move_batch_to_device, prune_checkpoints, save_json, set_seed
+from ..utils.common import (
+    count_parameters,
+    ensure_dir,
+    get_device,
+    load_torch_checkpoint,
+    move_batch_to_device,
+    prune_checkpoints,
+    save_json,
+    set_seed,
+)
@@
     def _step_scheduler(self, metric_value: float | None = None) -> None:
         scheduler = self.scheduler_bundle.scheduler
         if scheduler is None or self.scheduler_bundle.step_on == 'batch':
             return
@@
         else:
             scheduler.step()
+
+    def _restore_checkpoint(self, checkpoint_name: str) -> bool:
+        checkpoint_path = self.checkpoint_dir / checkpoint_name
+        if not checkpoint_path.exists():
+            return False
+        checkpoint = load_torch_checkpoint(checkpoint_path)
+        self.model.load_state_dict(checkpoint['model_state_dict'])
+        self.loss_bundle.load_state_dict(checkpoint.get('loss_bundle_state_dict', {}))
+        self.balancer.load_state_dict(checkpoint.get('balancer_state_dict', {}))
+        return True
@@
-        if self.test_loader is not None:
+        restored_best = self._restore_checkpoint('best.pt')
+        if restored_best and self.val_loader is not None:
+            final_metrics.update(self.evaluate(self.val_loader, split_name='best_val'))
+
+        if self.test_loader is not None:
             test_metrics = self.evaluate(self.test_loader, split_name='test')
             final_metrics.update(test_metrics)
```

### Expected effect

- `test/*` metrics reflect the model you actually selected.
- `best_val/*` gives you a sanity check that the restored checkpoint is the one you intended to keep.

---

## Patch 2: fix masked loss reduction so vector loss is not silently scaled by output dimension

### Why

`BaseTaskLoss._apply_sample_weight()` currently divides by `weight.sum()`, even when the loss tensor is shaped like `[B, D]` and the weights are only per-sample. That makes the effective task scale depend on `D`.

For `vector_regression`, that means the task is implicitly upweighted by the vector dimensionality unless you provide element-wise weights. That is almost never what you want for a multi-task setup with static weights.

### Files

- `mtl_mlp/training/losses.py`

### Diff

```diff
--- a/mtl_mlp/training/losses.py
+++ b/mtl_mlp/training/losses.py
@@
     @staticmethod
     def _apply_sample_weight(loss_tensor: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
         if sample_weight is None:
             return loss_tensor.mean()
-        weight = sample_weight
+        weight = sample_weight.to(device=loss_tensor.device, dtype=loss_tensor.dtype)
+
+        # If the weights are per-sample but the loss has extra feature dims,
+        # reduce to a per-sample mean first so the task scale does not depend
+        # on output dimensionality.
+        if weight.ndim == 1 and loss_tensor.ndim > 1 and weight.shape[0] == loss_tensor.shape[0]:
+            per_sample_loss = loss_tensor.reshape(loss_tensor.shape[0], -1).mean(dim=1)
+            return (per_sample_loss * weight).sum() / weight.sum().clamp_min(1e-8)
+
         while weight.ndim < loss_tensor.ndim:
             weight = weight.unsqueeze(-1)
         weighted = loss_tensor * weight
         normalizer = weight.sum().clamp_min(1e-8)
         return weighted.sum() / normalizer
```

### Expected effect

- Static weights become much easier to reason about.
- `vector_regression` no longer gets a hidden scaling bonus just because it has more output elements.
- Kendall-Gal / GradNorm also become easier to interpret because the raw task losses are more comparable.

---

## Patch 3: apply BC masks to BC metrics, and make control ratios sane when the baseline error is near zero

### Why

This file currently has two distinct problems:

1. **BC metrics ignore `bc_sample_weight` / `sample_weight`**, even though the config already routes BC masks in through `masks/bc_target_weight`.
2. **Control ratios explode** when the control baseline error is effectively zero.

Both problems can make the printed metrics actively misleading.

### Files

- `mtl_mlp/training/epoch_metrics.py`
- `mtl_mlp/training/trainer.py`

### Diff

```diff
--- a/mtl_mlp/training/epoch_metrics.py
+++ b/mtl_mlp/training/epoch_metrics.py
@@
 @dataclass
 class EpochAccumulator:
     bc_threshold: float = 0.5
     control_enabled: bool = False
     control_ratio_eps: float = 1.0e-8
     control_ratio_floor_quantile: float = 0.10
+    control_min_baseline_error: float = 1.0e-4
@@
-        self.bc_correct = 0
-        self.bc_tp = 0
-        self.bc_fp = 0
-        self.bc_fn = 0
-        self.bc_tn = 0
+        self.bc_correct = 0.0
+        self.bc_weight_total = 0.0
+        self.bc_tp = 0.0
+        self.bc_fp = 0.0
+        self.bc_fn = 0.0
+        self.bc_tn = 0.0
@@
         self.vector_control_ratios: list[torch.Tensor] = []
         self.reg_control_ratios: list[torch.Tensor] = []
+        self.vector_control_wins = 0.0
+        self.reg_control_wins = 0.0
+        self.vector_control_valid = 0.0
+        self.reg_control_valid = 0.0
+        self.vector_control_skipped = 0.0
+        self.reg_control_skipped = 0.0
@@
     def update_control(
@@
         if bool(torch.any(vec_mask)):
             vec_err = vec_err[vec_mask]
             vec_control_err = vec_control_err[vec_mask]
-            vec_floor = torch.clamp(
-                torch.quantile(vec_control_err, min(max(self.control_ratio_floor_quantile, 0.0), 1.0)),
-                min=self.control_ratio_eps,
-            )
-            vec_ratio = torch.abs(vec_err / torch.clamp(vec_control_err, min=vec_floor))
-            self.control_vec_abs += float(vec_control_err.sum().item())
-            self.control_vec_count += float(vec_err.numel())
-            self.vector_control_ratios.append(vec_ratio.detach().cpu())
+            vec_valid = vec_control_err > max(self.control_ratio_eps, self.control_min_baseline_error)
+            self.vector_control_skipped += float((~vec_valid).sum().item())
+            if bool(torch.any(vec_valid)):
+                vec_err = vec_err[vec_valid]
+                vec_control_err = vec_control_err[vec_valid]
+                vec_floor = torch.clamp(
+                    torch.quantile(vec_control_err, min(max(self.control_ratio_floor_quantile, 0.0), 1.0)),
+                    min=max(self.control_ratio_eps, self.control_min_baseline_error),
+                )
+                vec_ratio = torch.abs(vec_err / torch.clamp(vec_control_err, min=vec_floor))
+                self.control_vec_abs += float(vec_control_err.sum().item())
+                self.control_vec_count += float(vec_err.numel())
+                self.vector_control_wins += float((vec_err < vec_control_err).sum().item())
+                self.vector_control_valid += float(vec_err.numel())
+                self.vector_control_ratios.append(vec_ratio.detach().cpu())
@@
         if bool(torch.any(reg_mask)):
             reg_err = reg_err[reg_mask]
             reg_control_err = reg_control_err[reg_mask]
-            reg_floor = torch.clamp(
-                torch.quantile(reg_control_err, min(max(self.control_ratio_floor_quantile, 0.0), 1.0)),
-                min=self.control_ratio_eps,
-            )
-            reg_ratio = torch.abs(reg_err / torch.clamp(reg_control_err, min=reg_floor))
-            self.control_reg_abs += float(reg_control_err.sum().item())
-            self.control_reg_count += float(reg_err.numel())
-            self.reg_control_ratios.append(reg_ratio.detach().cpu())
+            reg_valid = reg_control_err > max(self.control_ratio_eps, self.control_min_baseline_error)
+            self.reg_control_skipped += float((~reg_valid).sum().item())
+            if bool(torch.any(reg_valid)):
+                reg_err = reg_err[reg_valid]
+                reg_control_err = reg_control_err[reg_valid]
+                reg_floor = torch.clamp(
+                    torch.quantile(reg_control_err, min(max(self.control_ratio_floor_quantile, 0.0), 1.0)),
+                    min=max(self.control_ratio_eps, self.control_min_baseline_error),
+                )
+                reg_ratio = torch.abs(reg_err / torch.clamp(reg_control_err, min=reg_floor))
+                self.control_reg_abs += float(reg_control_err.sum().item())
+                self.control_reg_count += float(reg_err.numel())
+                self.reg_control_wins += float((reg_err < reg_control_err).sum().item())
+                self.reg_control_valid += float(reg_err.numel())
+                self.reg_control_ratios.append(reg_ratio.detach().cpu())
@@
     def update_outputs(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
         bc_logits = outputs['bc'].detach()
         bc_target = batch['bc_target'].detach()
+
+        bc_ref = bc_logits.reshape(bc_logits.shape[0], -1)
+        bc_weight = self._task_weight(batch, 'bc_sample_weight', bc_ref)
+        bc_valid = bc_weight > 0.0
+
         if bc_logits.shape[-1] == 1:
-            probs = torch.sigmoid(bc_logits.reshape_as(bc_target))
-            preds = (probs >= self.bc_threshold).long()
-            targets = bc_target.long()
-            self.bc_correct += int((preds == targets).sum().item())
-            self.bc_tp += int(((preds == 1) & (targets == 1)).sum().item())
-            self.bc_fp += int(((preds == 1) & (targets == 0)).sum().item())
-            self.bc_fn += int(((preds == 0) & (targets == 1)).sum().item())
-            self.bc_tn += int(((preds == 0) & (targets == 0)).sum().item())
+            probs = torch.sigmoid(bc_logits.reshape_as(bc_target)).reshape(bc_target.shape[0], -1)[:, 0]
+            targets = bc_target.reshape(bc_target.shape[0], -1)[:, 0].long()
+            if bool(torch.any(bc_valid)):
+                preds = (probs >= self.bc_threshold).long()[bc_valid]
+                targets = targets[bc_valid]
+                weights = bc_weight[bc_valid]
+                self.bc_correct += float(((preds == targets).float() * weights).sum().item())
+                self.bc_weight_total += float(weights.sum().item())
+                self.bc_tp += float((((preds == 1) & (targets == 1)).float() * weights).sum().item())
+                self.bc_fp += float((((preds == 1) & (targets == 0)).float() * weights).sum().item())
+                self.bc_fn += float((((preds == 0) & (targets == 1)).float() * weights).sum().item())
+                self.bc_tn += float((((preds == 0) & (targets == 0)).float() * weights).sum().item())
         else:
-            preds = torch.argmax(bc_logits, dim=-1)
-            targets = bc_target.view(-1).long()
-            self.bc_correct += int((preds == targets).sum().item())
+            preds = torch.argmax(bc_logits, dim=-1)
+            targets = bc_target.view(-1).long()
+            if bool(torch.any(bc_valid)):
+                preds = preds[bc_valid]
+                targets = targets[bc_valid]
+                weights = bc_weight[bc_valid]
+                self.bc_correct += float(((preds == targets).float() * weights).sum().item())
+                self.bc_weight_total += float(weights.sum().item())
@@
-        total_bc = self.bc_tp + self.bc_fp + self.bc_fn + self.bc_tn
-        if total_bc > 0:
+        if self.bc_weight_total > 0:
+            metrics[f'{prefix}/bc_accuracy'] = self.bc_correct / max(self.bc_weight_total, 1.0)
+
+        total_bc = self.bc_tp + self.bc_fp + self.bc_fn + self.bc_tn
+        if total_bc > 0:
             precision = self.bc_tp / max(self.bc_tp + self.bc_fp, 1)
             recall = self.bc_tp / max(self.bc_tp + self.bc_fn, 1)
             f1 = 2 * precision * recall / max(precision + recall, 1e-8)
-            metrics[f'{prefix}/bc_accuracy'] = self.bc_correct / max(total_bc, 1)
             metrics[f'{prefix}/bc_precision'] = precision
             metrics[f'{prefix}/bc_recall'] = recall
             metrics[f'{prefix}/bc_f1'] = f1
@@
         if self.control_enabled:
             vec_stats = self._summarize_ratio(self.vector_control_ratios)
             reg_stats = self._summarize_ratio(self.reg_control_ratios)
             metrics[f'{prefix}/control_vector_mae'] = self.control_vec_abs / max(self.control_vec_count, 1)
             metrics[f'{prefix}/control_reg_mae'] = self.control_reg_abs / max(self.control_reg_count, 1)
             metrics[f'{prefix}/control_growth_mae'] = metrics[f'{prefix}/control_reg_mae']
+            metrics[f'{prefix}/vector_vs_control_win_rate'] = self.vector_control_wins / max(self.vector_control_valid, 1.0)
+            metrics[f'{prefix}/reg_vs_control_win_rate'] = self.reg_control_wins / max(self.reg_control_valid, 1.0)
+            metrics[f'{prefix}/vector_vs_control_skipped_frac'] = self.vector_control_skipped / max(self.vector_control_valid + self.vector_control_skipped, 1.0)
+            metrics[f'{prefix}/reg_vs_control_skipped_frac'] = self.reg_control_skipped / max(self.reg_control_valid + self.reg_control_skipped, 1.0)
             metrics[f'{prefix}/vector_vs_control_frac_mean'] = vec_stats['mean']
             metrics[f'{prefix}/vector_vs_control_frac_median'] = vec_stats['median']
             metrics[f'{prefix}/vector_vs_control_frac_p95'] = vec_stats['p95']
             metrics[f'{prefix}/vector_vs_control_frac_p99'] = vec_stats['p99']
             metrics[f'{prefix}/reg_vs_control_frac_mean'] = reg_stats['mean']
```

And wire the new config field through the trainer:

```diff
--- a/mtl_mlp/training/trainer.py
+++ b/mtl_mlp/training/trainer.py
@@
         self.control_ratio_eps = float(config.evaluation.get_path('control.ratio_eps', 1.0e-8))
         self.control_ratio_floor_quantile = float(config.evaluation.get_path('control.ratio_floor_quantile', 0.10))
+        self.control_min_baseline_error = float(config.evaluation.get_path('control.min_baseline_error', 1.0e-4))
@@
     def _build_accumulator(self, control_enabled: bool) -> EpochAccumulator:
         return EpochAccumulator(
             bc_threshold=self.bc_threshold,
             control_enabled=control_enabled,
             control_ratio_eps=self.control_ratio_eps,
             control_ratio_floor_quantile=self.control_ratio_floor_quantile,
+            control_min_baseline_error=self.control_min_baseline_error,
         )
```

### Expected effect

- BC metrics finally agree with the supervision mask that the loss is already using.
- Control comparisons stop producing comic-book nonsense when the baseline error is zero.
- You get a more useful `*_vs_control_win_rate` summary, which is easier to reason about than ratio explosions.

---

## Patch 4: fix the V100 sweep so batch size is selected by validation quality, not by a fake throughput contest

### Why

Right now the sweep is doing something dumb:

- batch phase runs with `val_steps=0`
- then it picks the batch size using only `samples_per_sec`

But your own sweep results say throughput is basically identical from BS 256 through BS 8192. So the current selector is pretending to optimize something it is not actually learning anything from.

### Files

- `scripts/v100_tune_sweep.py`
- `Makefile`

### Diff

```diff
--- a/scripts/v100_tune_sweep.py
+++ b/scripts/v100_tune_sweep.py
@@
 def _rank_key(entry: dict[str, Any]) -> tuple[float, float]:
     # lower val loss is better; among ties, prefer higher throughput.
     val = entry.get("val_loss_mean", float("nan"))
     if not math.isfinite(val):
         val = float("inf")
     sps = float(entry.get("samples_per_sec", 0.0))
     return (val, -sps)
+
+
+def _select_batch_size(batch_results: list[dict[str, Any]], rel_tol: float = 0.05) -> int:
+    ranked = [
+        entry
+        for entry in batch_results
+        if entry.get("status") == "ok" and math.isfinite(float(entry.get("val_loss_mean", float("nan"))))
+    ]
+    if not ranked:
+        ranked = [entry for entry in batch_results if entry.get("status") == "ok"]
+        if not ranked:
+            raise ValueError("No successful batch-size trials.")
+        return max(ranked, key=lambda entry: float(entry.get("samples_per_sec", 0.0)))["batch_size"]
+
+    ranked.sort(key=_rank_key)
+    best_val = float(ranked[0]["val_loss_mean"])
+    near_best = [entry for entry in ranked if float(entry["val_loss_mean"]) <= best_val * (1.0 + rel_tol)]
+
+    # Throughput is almost flat in the current sweep, so among near-ties,
+    # prefer the smaller batch size for better optimizer granularity.
+    return min(near_best, key=lambda entry: int(entry["batch_size"]))["batch_size"]
@@
     parser.add_argument("--hyper_train_steps", type=int, default=200)
     parser.add_argument("--hyper_sample_budget", type=int, default=65536)
+    parser.add_argument("--batch_val_steps", type=int, default=10)
     parser.add_argument("--val_steps", type=int, default=40)
+    parser.add_argument("--batch_selection_rel_tol", type=float, default=0.05)
     parser.add_argument("--selected_batch_size", type=int, default=None)
@@
-        result = _run_trial(cfg, train_steps=train_steps, val_steps=0)
+        result = _run_trial(cfg, train_steps=train_steps, val_steps=args.batch_val_steps)
@@
     if args.selected_batch_size is not None:
         chosen_batch = int(args.selected_batch_size)
     elif valid_batch:
-        chosen_batch = max(valid_batch, key=lambda entry: float(entry.get("samples_per_sec", 0.0)))["batch_size"]
+        chosen_batch = _select_batch_size(valid_batch, rel_tol=float(args.batch_selection_rel_tol))
     else:
         chosen_batch = min(batch_sizes)
```

```diff
--- a/Makefile
+++ b/Makefile
@@
 SWEEP_BATCH_TRAIN_STEPS ?= 150
 SWEEP_BATCH_SAMPLE_BUDGET ?= 65536
+SWEEP_BATCH_VAL_STEPS ?= 10
 SWEEP_HYPER_TRAIN_STEPS ?= 200
 SWEEP_HYPER_SAMPLE_BUDGET ?= 65536
 SWEEP_VAL_STEPS ?= 40
+SWEEP_BATCH_SELECTION_REL_TOL ?= 0.05
@@
 		--batch_train_steps "$(SWEEP_BATCH_TRAIN_STEPS)" \
 		--batch_sample_budget "$(SWEEP_BATCH_SAMPLE_BUDGET)" \
+		--batch_val_steps "$(SWEEP_BATCH_VAL_STEPS)" \
 		--hyper_train_steps "$(SWEEP_HYPER_TRAIN_STEPS)" \
 		--hyper_sample_budget "$(SWEEP_HYPER_SAMPLE_BUDGET)" \
+		--batch_selection_rel_tol "$(SWEEP_BATCH_SELECTION_REL_TOL)" \
 		--val_steps "$(SWEEP_VAL_STEPS)" ); \
```

### Expected effect

- Batch-size choice becomes driven by actual validation behavior.
- Given the current sweep results, this will probably push you toward **256 or 512**, not `1024`, unless a real val pass proves otherwise.

---

## Patch 5: add a contiguous-block sampler for the huge unchunked HDF5 files

### Why

Your HDF5 files are mostly **not chunked** and **not compressed**. Random single-sample access is the wrong access pattern for that storage layout.

The batch sweep throughput being flat across batch sizes is exactly what I would expect if the pipeline is I/O-bound on scattered reads. You want to preserve locality within each file and shuffle at the **block** level, not at the individual sample level.

### Files

- `mtl_mlp/data/hdf5_dataset.py`
- `mtl_mlp/data/samplers.py` (new)
- `configs/rhea_v100_recommended.yaml`
- `configs/rhea_v100_recommended_full_template.yaml`
- `configs/rhea_v100_sweep.yaml`

### New file

```diff
--- /dev/null
+++ b/mtl_mlp/data/samplers.py
@@
+from __future__ import annotations
+
+import math
+import random
+from typing import Iterator
+
+from torch.utils.data import Sampler
+
+
+class ContiguousBlockBatchSampler(Sampler[list[int]]):
+    def __init__(
+        self,
+        dataset,
+        batch_size: int,
+        block_size: int | None = None,
+        shuffle: bool = True,
+        drop_last: bool = False,
+        seed: int = 42,
+    ) -> None:
+        self.dataset = dataset
+        self.batch_size = int(batch_size)
+        self.block_size = int(block_size or max(self.batch_size * 8, 2048))
+        self.shuffle = bool(shuffle)
+        self.drop_last = bool(drop_last)
+        self.seed = int(seed)
+
+    def _build_blocks(self) -> list[tuple[int, int]]:
+        blocks: list[tuple[int, int]] = []
+        start = 0
+        for file_len in self.dataset._lengths:
+            file_start = start
+            file_end = start + int(file_len)
+            for block_start in range(file_start, file_end, self.block_size):
+                blocks.append((block_start, min(block_start + self.block_size, file_end)))
+            start = file_end
+        return blocks
+
+    def __iter__(self) -> Iterator[list[int]]:
+        rng = random.Random(self.seed)
+        blocks = self._build_blocks()
+        if self.shuffle:
+            rng.shuffle(blocks)
+
+        batch: list[int] = []
+        for start, end in blocks:
+            for idx in range(start, end):
+                batch.append(idx)
+                if len(batch) == self.batch_size:
+                    yield batch
+                    batch = []
+
+        if batch and not self.drop_last:
+            yield batch
+
+    def __len__(self) -> int:
+        if self.drop_last:
+            return len(self.dataset) // self.batch_size
+        return math.ceil(len(self.dataset) / self.batch_size)
```

### Existing file diff

```diff
--- a/mtl_mlp/data/hdf5_dataset.py
+++ b/mtl_mlp/data/hdf5_dataset.py
@@
-from torch.utils.data import DataLoader, Dataset
+from torch.utils.data import DataLoader, Dataset
+
+from .samplers import ContiguousBlockBatchSampler
@@
 def build_dataloader(dataset: Dataset | None, config: Any, train: bool) -> DataLoader | None:
     if dataset is None:
         return None
     loader_cfg = config.data.loader
     num_workers = int(loader_cfg.get('num_workers', 0))
-    return DataLoader(
-        dataset,
-        batch_size=int(loader_cfg.get('batch_size', 256)),
-        shuffle=bool(loader_cfg.get('shuffle_train', True)) if train else False,
-        num_workers=num_workers,
-        pin_memory=bool(loader_cfg.get('pin_memory', True)),
-        persistent_workers=bool(loader_cfg.get('persistent_workers', False)) if num_workers > 0 else False,
-        drop_last=bool(loader_cfg.get('drop_last', False)) if train else False,
-    )
+    batch_size = int(loader_cfg.get('batch_size', 256))
+    common_kwargs: dict[str, Any] = {
+        'num_workers': num_workers,
+        'pin_memory': bool(loader_cfg.get('pin_memory', True)),
+        'persistent_workers': bool(loader_cfg.get('persistent_workers', False)) if num_workers > 0 else False,
+    }
+    if num_workers > 0 and loader_cfg.get('prefetch_factor') is not None:
+        common_kwargs['prefetch_factor'] = int(loader_cfg.get('prefetch_factor'))
+
+    sampler_cfg = loader_cfg.get('sampler', {'name': 'default'})
+    sampler_name = str(sampler_cfg.get('name', 'default')).lower()
+    if train and sampler_name == 'contiguous_blocks':
+        batch_sampler = ContiguousBlockBatchSampler(
+            dataset,
+            batch_size=batch_size,
+            block_size=int(sampler_cfg.get('block_size', max(batch_size * 8, 2048))),
+            shuffle=bool(loader_cfg.get('shuffle_train', True)),
+            drop_last=bool(loader_cfg.get('drop_last', False)),
+            seed=int(config.get('seed', 42)),
+        )
+        return DataLoader(dataset, batch_sampler=batch_sampler, **common_kwargs)
+
+    return DataLoader(
+        dataset,
+        batch_size=batch_size,
+        shuffle=bool(loader_cfg.get('shuffle_train', True)) if train else False,
+        drop_last=bool(loader_cfg.get('drop_last', False)) if train else False,
+        **common_kwargs,
+    )
```

### Config diffs

```diff
--- a/configs/rhea_v100_recommended.yaml
+++ b/configs/rhea_v100_recommended.yaml
@@
   loader:
     batch_size: 1024
     num_workers: 4
     pin_memory: true
     persistent_workers: true
+    prefetch_factor: 4
     drop_last: false
     shuffle_train: true
+    sampler:
+      name: contiguous_blocks
+      block_size: 8192
```

```diff
--- a/configs/rhea_v100_recommended_full_template.yaml
+++ b/configs/rhea_v100_recommended_full_template.yaml
@@
   loader:
     batch_size: 1024
     num_workers: 4
     pin_memory: true
     persistent_workers: true
+    prefetch_factor: 4
     drop_last: false
     shuffle_train: true
+    sampler:
+      name: contiguous_blocks
+      block_size: 8192
```

```diff
--- a/configs/rhea_v100_sweep.yaml
+++ b/configs/rhea_v100_sweep.yaml
@@
   loader:
     batch_size: 1024
     num_workers: 4
     pin_memory: true
     persistent_workers: true
+    prefetch_factor: 4
     drop_last: false
     shuffle_train: true
+    sampler:
+      name: contiguous_blocks
+      block_size: 8192
```

### Expected effect

- Better I/O locality on your unchunked HDF5 files.
- More realistic batch-size tuning because the loader stops sabotaging the experiment.
- This is the first place I would spend performance effort before touching model math.

---

## Patch 6: add a per-file evaluation report, because your dataset is too heterogeneous for a single global mean

### Why

Your file families are wildly different in size and content. A single global mean can look acceptable while one or two families are failing completely.

I would add a per-file report **before** doing more model sweeps. Otherwise you are flying blind and calling it science.

### Files

- `mtl_mlp/training/trainer.py`
- `evaluate.py`

### Suggested implementation

This one is a little larger, so I am giving the implementation shape rather than a line-perfect patch.

### Diff skeleton

```diff
--- a/mtl_mlp/training/trainer.py
+++ b/mtl_mlp/training/trainer.py
@@
 class Trainer:
@@
         self.metrics_path = self.run_dir / 'final_metrics.json'
+        self.metrics_by_file_path = self.run_dir / 'final_metrics_by_file.json'
         self.config_snapshot_path = self.run_dir / 'config_snapshot.json'
@@
     def evaluate(self, loader: Any | None = None, split_name: str = 'eval') -> dict[str, float]:
         loader = loader or self.val_loader or self.test_loader
         if loader is None:
             return {}
         self.model.train(False)
         accumulator = self._build_accumulator(control_enabled=self.control_enabled)
+        per_file: dict[int, EpochAccumulator] = {}
         progress = tqdm(loader, desc=split_name, leave=False)
         for raw_batch in progress:
             batch = move_batch_to_device(raw_batch, self.device)
             outputs, detached_log = self._evaluate_batch(batch)
             batch_size = int(batch['inputs'].shape[0])
             accumulator.update_losses(detached_log, batch_size)
             accumulator.update_outputs(outputs, batch)
+
+            file_indices = batch['file_index']
+            unique_files = torch.unique(file_indices)
+            for file_index in unique_files.tolist():
+                mask = file_indices == file_index
+                sub_outputs = {
+                    key: value[mask] for key, value in outputs.items()
+                }
+                sub_batch = {
+                    key: (value[mask] if isinstance(value, torch.Tensor) and value.shape[0] == mask.shape[0] else value)
+                    for key, value in batch.items()
+                }
+                acc = per_file.setdefault(file_index, self._build_accumulator(control_enabled=False))
+                acc.update_outputs(sub_outputs, sub_batch)
+
             if self.control_enabled:
                 control_vector, control_reg = self._compute_control_baseline(batch['inputs'])
                 accumulator.update_control(outputs, batch, control_vector, control_reg)
-        return accumulator.summarize(split_name)
+        metrics = accumulator.summarize(split_name)
+        dataset = getattr(loader, 'dataset', None)
+        files = getattr(dataset, 'files', []) if dataset is not None else []
+        metrics_by_file = {}
+        for file_index, acc in per_file.items():
+            file_name = files[file_index] if file_index < len(files) else str(file_index)
+            metrics_by_file[file_name] = acc.summarize(split_name)
+        self.last_metrics_by_file = metrics_by_file
+        return metrics
```

```diff
--- a/evaluate.py
+++ b/evaluate.py
@@
     metrics = trainer.evaluate(loader=loaders[args.split], split_name=args.split)
     if args.output:
         save_json(metrics, args.output)
+        if getattr(trainer, 'last_metrics_by_file', None):
+            per_file_output = args.output.replace('.json', '_by_file.json')
+            save_json(trainer.last_metrics_by_file, per_file_output)
     print(json.dumps(metrics, indent=2, sort_keys=True))
```

### Expected effect

- You will see immediately whether a sweep improvement is broad or just concentrated on one easy family.
- This is especially important before deciding whether the split policy is actually robust.

---

## Immediate config stance after the patches

### Keep

- **large model preset** as the starting point
- **batch norm off**
- **AdamW + cosine** as the first-line baseline

### Re-test after patches

- **batch size**: do not trust the current `1024` recommendation until Patch 4 is in place
- **best checkpoint test metrics**: re-run once Patch 1 is in
- **BC metrics**: re-check after Patch 3, because the current numbers may be inflated or distorted by masked samples being counted anyway

### Do not spend time on yet

- adding more exotic trunks
- changing the head architecture
- fiddling with PCGrad / GradNorm before you fix the evaluation and data path

If you only implement **three** things first, do these:

1. Patch 1
2. Patch 3
3. Patch 4

That will immediately make the repo more honest, which is rare and therefore valuable.
