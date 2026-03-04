from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += int(n)

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)



def ensure_dir(path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target



def save_json(payload: dict[str, Any], path: str | os.PathLike[str]) -> None:
    with Path(path).open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)



def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def get_device(device_name: str = 'auto') -> torch.device:
    name = str(device_name).lower()
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def configure_torch_runtime(num_threads: int | None = None, num_interop_threads: int | None = None) -> None:
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))
    if num_interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(num_interop_threads))
        except RuntimeError:
            # PyTorch only lets you set this once per process.
            pass



def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        elif isinstance(value, dict):
            moved[key] = move_batch_to_device(value, device)
        else:
            moved[key] = value
    return moved



def module_from_path(model: nn.Module, dotted_path: str) -> nn.Module:
    module: nn.Module = model
    for part in dotted_path.split('.'):
        if isinstance(module, nn.ModuleDict):
            module = module[part]
        else:
            module = getattr(module, part)
    return module



def freeze_module(module: nn.Module, freeze_batch_norm_stats: bool = True) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False
    if freeze_batch_norm_stats:
        for submodule in module.modules():
            if isinstance(submodule, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                submodule.eval()



def count_parameters(module: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return {'total': int(total), 'trainable': int(trainable)}



def prune_checkpoints(directory: str | os.PathLike[str], keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    checkpoint_dir = Path(directory)
    checkpoints = sorted(checkpoint_dir.glob('epoch_*.pt'))
    if len(checkpoints) <= keep_last_n:
        return
    for path in checkpoints[:-keep_last_n]:
        path.unlink(missing_ok=True)
