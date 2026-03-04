from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from mtl_mlp.config import ConfigNode, load_config, validate_config
from mtl_mlp.data import build_dataloader, build_datasets
from mtl_mlp.models import MultiTaskMLP
from mtl_mlp.training import build_loss_bundle
from mtl_mlp.training.trainer import Trainer
from mtl_mlp.utils import configure_torch_runtime, move_batch_to_device, set_seed


MODEL_PRESETS: dict[str, dict[str, list[int] | float]] = {
    "small": {
        "trunk": [128, 64],
        "bc": [32],
        "vector": [64],
        "reg": [32],
        "dropout": 0.05,
    },
    "medium": {
        "trunk": [256, 256, 128],
        "bc": [64],
        "vector": [128, 64],
        "reg": [64],
        "dropout": 0.10,
    },
    "large": {
        "trunk": [512, 512, 256],
        "bc": [128, 64],
        "vector": [256, 128],
        "reg": [128, 64],
        "dropout": 0.10,
    },
}


def _cleanup_trainer(trainer: Trainer, loaders: list[Any]) -> None:
    for loader in loaders:
        if loader is None:
            continue
        dataset = getattr(loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "close"):
            dataset.close()
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _override_common(cfg: dict[str, Any]) -> None:
    # Keep sweeps focused on train/val behavior; test is not needed in tuning loop.
    cfg["data"]["test_files"] = []
    cfg["training"]["device"] = "auto"
    cfg["training"]["epochs"] = 1
    cfg["training"]["cpu_num_threads"] = 8
    cfg["training"]["cpu_num_interop_threads"] = 2
    cfg["training"]["deterministic"] = False
    cfg["training"]["compile_model"] = False
    cfg["training"]["gradient_accumulation_steps"] = 1
    cfg["training"]["mixed_precision"] = True
    cfg["training"]["scheduler"]["name"] = "none"
    cfg["training"]["early_stopping"]["enabled"] = False
    cfg["logging"]["save_every_epoch"] = False
    cfg["logging"]["keep_last_n_checkpoints"] = 1
    cfg["data"]["loader"]["num_workers"] = 0
    cfg["data"]["loader"]["persistent_workers"] = False
    cfg["data"]["loader"]["pin_memory"] = True


def _apply_model_preset(cfg: dict[str, Any], preset_name: str, batch_norm: bool) -> None:
    preset = MODEL_PRESETS[preset_name]
    cfg["model"]["trunk"]["hidden_dims"] = list(preset["trunk"])
    cfg["model"]["trunk"]["dropout"] = float(preset["dropout"])
    cfg["model"]["trunk"]["batch_norm"] = bool(batch_norm)
    cfg["model"]["heads"]["bc"]["hidden_dims"] = list(preset["bc"])
    cfg["model"]["heads"]["bc"]["batch_norm"] = bool(batch_norm)
    cfg["model"]["heads"]["vector_regression"]["hidden_dims"] = list(preset["vector"])
    cfg["model"]["heads"]["vector_regression"]["batch_norm"] = bool(batch_norm)
    cfg["model"]["heads"]["regression"]["hidden_dims"] = list(preset["reg"])
    cfg["model"]["heads"]["regression"]["batch_norm"] = bool(batch_norm)


def _set_optimizer_lr(cfg: dict[str, Any], lr: float) -> None:
    cfg["training"]["optimizer"]["lr"] = float(lr)
    for group in cfg["training"]["optimizer"].get("param_groups", []):
        group["lr"] = float(lr)


def _build_trainer(cfg_dict: dict[str, Any]) -> tuple[Trainer, Any, Any]:
    config = ConfigNode(cfg_dict)
    validate_config(config)
    set_seed(int(config.get("seed", 42)), deterministic=bool(config.training.get("deterministic", False)))

    datasets = build_datasets(config)
    train_loader = build_dataloader(datasets["train"], config, train=True)
    val_loader = build_dataloader(datasets["val"], config, train=False)

    model = MultiTaskMLP(config)
    loss_bundle = build_loss_bundle(config)
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        loss_bundle=loss_bundle,
    )
    return trainer, train_loader, val_loader


def _run_trial(
    cfg_dict: dict[str, Any],
    train_steps: int,
    val_steps: int,
) -> dict[str, Any]:
    trainer: Trainer | None = None
    train_loader = None
    val_loader = None
    try:
        trainer, train_loader, val_loader = _build_trainer(cfg_dict)
        if train_loader is None:
            raise ValueError("No train loader available.")

        device = trainer.device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        trainer.model.train(True)
        train_loss_total = 0.0
        train_batches = 0
        train_samples = 0
        t0 = time.perf_counter()
        for step_idx, raw_batch in enumerate(train_loader):
            batch = move_batch_to_device(raw_batch, device)
            _, detached_log = trainer._train_batch(batch)
            batch_size = int(batch["inputs"].shape[0])
            train_loss_total += float(detached_log["total_loss"])
            train_batches += 1
            train_samples += batch_size
            if step_idx + 1 >= train_steps:
                break
        train_seconds = time.perf_counter() - t0

        val_loss_mean = float("nan")
        if val_loader is not None and val_steps > 0:
            trainer.model.train(False)
            val_loss_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for step_idx, raw_batch in enumerate(val_loader):
                    batch = move_batch_to_device(raw_batch, device)
                    _, detached_log = trainer._evaluate_batch(batch)
                    val_loss_total += float(detached_log["total_loss"])
                    val_batches += 1
                    if step_idx + 1 >= val_steps:
                        break
            if val_batches > 0:
                val_loss_mean = val_loss_total / val_batches

        peak_mem_gb = float("nan")
        if device.type == "cuda":
            peak_mem_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3))

        return {
            "status": "ok",
            "train_batches": train_batches,
            "train_samples": train_samples,
            "train_seconds": train_seconds,
            "samples_per_sec": (train_samples / max(train_seconds, 1.0e-9)),
            "steps_per_sec": (train_batches / max(train_seconds, 1.0e-9)),
            "train_loss_mean": (train_loss_total / max(train_batches, 1)),
            "val_loss_mean": val_loss_mean,
            "peak_mem_gb": peak_mem_gb,
            "train_dataset_size": len(train_loader.dataset) if train_loader is not None else 0,
            "val_dataset_size": len(val_loader.dataset) if val_loader is not None else 0,
        }
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"status": "oom", "error": message}
        return {"status": "error", "error": message}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    finally:
        if trainer is not None:
            _cleanup_trainer(trainer, [train_loader, val_loader])


def _rank_key(entry: dict[str, Any]) -> tuple[float, float]:
    # lower val loss is better; among ties, prefer higher throughput.
    val = entry.get("val_loss_mean", float("nan"))
    if not math.isfinite(val):
        val = float("inf")
    sps = float(entry.get("samples_per_sec", 0.0))
    return (val, -sps)


def main() -> None:
    parser = argparse.ArgumentParser(description="V100 batch-size and hyperparameter tuning sweep.")
    parser.add_argument("--base_config", default="configs/rhea_v100_sweep.yaml")
    parser.add_argument("--output_json", default="outputs/tuning/v100_sweep_results.json")
    parser.add_argument("--output_csv", default="outputs/tuning/v100_sweep_results.csv")
    parser.add_argument("--batch_sizes", default="256,512,1024,2048,4096,8192")
    parser.add_argument("--learning_rates", default="3e-4,1e-3,3e-3")
    parser.add_argument("--model_sizes", default="small,medium,large")
    parser.add_argument("--batch_norm_options", default="0,1")
    parser.add_argument("--batch_train_steps", type=int, default=150)
    parser.add_argument("--batch_sample_budget", type=int, default=65536)
    parser.add_argument("--hyper_train_steps", type=int, default=200)
    parser.add_argument("--hyper_sample_budget", type=int, default=65536)
    parser.add_argument("--val_steps", type=int, default=40)
    parser.add_argument("--selected_batch_size", type=int, default=None)
    args = parser.parse_args()

    base_config = load_config(args.base_config).to_dict()
    configure_torch_runtime(
        num_threads=base_config["training"].get("cpu_num_threads"),
        num_interop_threads=base_config["training"].get("cpu_num_interop_threads"),
    )
    out_json = Path(args.output_json).expanduser().resolve()
    out_csv = Path(args.output_csv).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    batch_sizes = [int(item.strip()) for item in args.batch_sizes.split(",") if item.strip()]
    lrs = [float(item.strip()) for item in args.learning_rates.split(",") if item.strip()]
    model_sizes = [item.strip() for item in args.model_sizes.split(",") if item.strip()]
    bn_opts = [bool(int(item.strip())) for item in args.batch_norm_options.split(",") if item.strip()]

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "cpu-only"
    print(f"[env] gpu={gpu_name}")

    batch_results: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        cfg = copy.deepcopy(base_config)
        _override_common(cfg)
        cfg["output"]["experiment_name"] = f"v100_batch_bs{batch_size}"
        cfg["data"]["loader"]["batch_size"] = batch_size
        _set_optimizer_lr(cfg, 1.0e-3)
        _apply_model_preset(cfg, preset_name="medium", batch_norm=False)
        budget_steps = int(math.ceil(float(max(args.batch_sample_budget, 1)) / float(max(batch_size, 1))))
        train_steps = max(4, min(int(max(args.batch_train_steps, 1)), budget_steps))
        print(f"[batch] bs={batch_size} train_steps={train_steps}")
        result = _run_trial(cfg, train_steps=train_steps, val_steps=0)
        result.update(
            {
                "phase": "batch",
                "batch_size": batch_size,
                "configured_train_steps": train_steps,
                "lr": 1.0e-3,
                "model_size": "medium",
                "batch_norm": False,
            }
        )
        batch_results.append(result)
        print(f"[batch-result] {result}")

    valid_batch = [entry for entry in batch_results if entry.get("status") == "ok"]
    if args.selected_batch_size is not None:
        chosen_batch = int(args.selected_batch_size)
    elif valid_batch:
        chosen_batch = max(valid_batch, key=lambda entry: float(entry.get("samples_per_sec", 0.0)))["batch_size"]
    else:
        chosen_batch = min(batch_sizes)
    print(f"[selection] chosen_batch_size={chosen_batch}")

    hyper_results: list[dict[str, Any]] = []
    for model_size, batch_norm, lr in itertools.product(model_sizes, bn_opts, lrs):
        cfg = copy.deepcopy(base_config)
        _override_common(cfg)
        cfg["output"]["experiment_name"] = (
            f"v100_hyper_{model_size}_bn{int(batch_norm)}_lr{lr:g}_bs{chosen_batch}"
        )
        cfg["data"]["loader"]["batch_size"] = int(chosen_batch)
        _set_optimizer_lr(cfg, float(lr))
        _apply_model_preset(cfg, preset_name=model_size, batch_norm=batch_norm)
        budget_steps = int(math.ceil(float(max(args.hyper_sample_budget, 1)) / float(max(chosen_batch, 1))))
        train_steps = max(4, min(int(max(args.hyper_train_steps, 1)), budget_steps))
        print(
            f"[hyper] model={model_size} bn={batch_norm} lr={lr} bs={chosen_batch} "
            f"train_steps={train_steps}"
        )
        result = _run_trial(cfg, train_steps=train_steps, val_steps=args.val_steps)
        result.update(
            {
                "phase": "hyper",
                "batch_size": int(chosen_batch),
                "configured_train_steps": train_steps,
                "lr": float(lr),
                "model_size": model_size,
                "batch_norm": bool(batch_norm),
            }
        )
        hyper_results.append(result)
        print(f"[hyper-result] {result}")

    ranked_hyper = [entry for entry in hyper_results if entry.get("status") == "ok"]
    ranked_hyper.sort(key=_rank_key)

    payload = {
        "base_config": str(Path(args.base_config).expanduser().resolve()),
        "gpu": gpu_name,
        "batch_results": batch_results,
        "chosen_batch_size": int(chosen_batch),
        "hyper_results": hyper_results,
        "hyper_ranked": ranked_hyper,
    }

    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)

    csv_fields = [
        "phase",
        "status",
        "batch_size",
        "lr",
        "model_size",
        "batch_norm",
        "train_samples",
        "train_batches",
        "train_seconds",
        "samples_per_sec",
        "steps_per_sec",
        "peak_mem_gb",
        "train_loss_mean",
        "val_loss_mean",
        "error",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in batch_results + hyper_results:
            writer.writerow({key: row.get(key) for key in csv_fields})

    print(f"[done] json={out_json}")
    print(f"[done] csv={out_csv}")
    if ranked_hyper:
        print("[best]", ranked_hyper[0])


if __name__ == "__main__":
    main()
