"""Configurable multi-task MLP pipeline."""

from .config import ConfigNode, load_config

__all__ = [
    "ConfigNode",
    "load_config",
    "MultiTaskMLP",
    "MultiFileHDF5Dataset",
    "Trainer",
]


def __getattr__(name: str):
    if name == "MultiTaskMLP":
        from .models.multitask_model import MultiTaskMLP

        return MultiTaskMLP
    if name == "MultiFileHDF5Dataset":
        from .data.hdf5_dataset import MultiFileHDF5Dataset

        return MultiFileHDF5Dataset
    if name == "Trainer":
        from .training.trainer import Trainer

        return Trainer
    raise AttributeError(f"module 'mtl_mlp' has no attribute {name!r}")
