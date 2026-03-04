from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .samplers import ContiguousBlockBatchSampler


@dataclass
class FileIndexEntry:
    file_index: int
    local_index: int


class MultiFileHDF5Dataset(Dataset):
    """Read multiple HDF5 files as one dataset with lazy file opening.

    Expected datasets per file are configurable and can live at nested paths
    like ``targets/bc``.
    """

    def __init__(
        self,
        files: list[str],
        key_map: dict[str, str | None],
        strict: bool = True,
        swmr: bool = False,
        input_dtype: torch.dtype = torch.float32,
        target_dtype: torch.dtype = torch.float32,
        require_targets: bool = True,
        reshape_map: dict[str, str] | None = None,
        input_feature_dim: int | None = 24,
    ) -> None:
        super().__init__()
        self.files = [str(Path(path).expanduser().resolve()) for path in files]
        self.key_map = key_map
        self.strict = strict
        self.swmr = swmr
        self.input_dtype = input_dtype
        self.target_dtype = target_dtype
        self.require_targets = require_targets
        default_reshape_map = {
            'input': 'flatten',
            'vector_target': 'flatten',
        }
        if reshape_map:
            default_reshape_map.update({str(key): str(value) for key, value in reshape_map.items()})
        self.reshape_map = default_reshape_map
        self.input_feature_dim = input_feature_dim
        self._handles: dict[int, h5py.File] = {}
        self._lengths: list[int] = []
        self._cumulative: list[int] = []
        self._validate_files()

    def _validate_files(self) -> None:
        if not self.files:
            self._lengths = []
            self._cumulative = []
            return

        required_keys = ['input'] if not self.require_targets else ['input', 'bc_target', 'vector_target', 'reg_target']
        strict_optional_keys = {
            'sample_weight',
            'bc_sample_weight',
            'vector_sample_weight',
            'reg_sample_weight',
        }
        running_total = 0
        cumulative: list[int] = []
        lengths: list[int] = []

        for file_path in self.files:
            if not Path(file_path).exists():
                raise FileNotFoundError(f'HDF5 file does not exist: {file_path}')
            with h5py.File(file_path, 'r', swmr=self.swmr, libver='latest') as handle:
                lengths_for_keys: list[int] = []
                for logical_key in required_keys:
                    hdf5_key = self.key_map.get(logical_key)
                    if not hdf5_key:
                        raise KeyError(f'Missing key map entry for {logical_key!r}.')
                    if hdf5_key not in handle:
                        raise KeyError(f"Dataset key '{hdf5_key}' not found in file {file_path}")
                    lengths_for_keys.append(int(handle[hdf5_key].shape[0]))
                for logical_key in ['sample_weight', 'bc_sample_weight', 'vector_sample_weight', 'reg_sample_weight']:
                    optional_key = self.key_map.get(logical_key)
                    if not optional_key:
                        continue
                    if optional_key in handle:
                        lengths_for_keys.append(int(handle[optional_key].shape[0]))
                    elif self.strict and logical_key in strict_optional_keys:
                        raise KeyError(f"Configured key '{optional_key}' ({logical_key}) not found in file {file_path}")
                unique_lengths = set(lengths_for_keys)
                if len(unique_lengths) != 1:
                    raise ValueError(
                        f'Found inconsistent leading dimensions in file {file_path}: {sorted(unique_lengths)}'
                    )
                file_len = lengths_for_keys[0]
                if self.strict:
                    input_shape = handle[self.key_map['input']].shape
                    if len(input_shape) < 2:
                        raise ValueError(
                            f"Expected input dataset '{self.key_map['input']}' to have at least 2 dimensions, got {input_shape}"
                        )
                    flattened_dim = int(np.prod(input_shape[1:]))
                    expected_dim = self.input_feature_dim
                    if expected_dim is not None and flattened_dim != expected_dim:
                        raise ValueError(
                            f"Expected input dataset '{self.key_map['input']}' to flatten to [N, {expected_dim}], "
                            f'got shape {input_shape}'
                        )
                lengths.append(file_len)
                running_total += file_len
                cumulative.append(running_total)

        self._lengths = lengths
        self._cumulative = cumulative

    def __len__(self) -> int:
        return self._cumulative[-1] if self._cumulative else 0

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state['_handles'] = {}
        return state

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles = {}

    def __del__(self) -> None:
        self.close()

    def _get_handle(self, file_index: int) -> h5py.File:
        if file_index not in self._handles:
            self._handles[file_index] = h5py.File(
                self.files[file_index],
                'r',
                swmr=self.swmr,
                libver='latest',
            )
        return self._handles[file_index]

    def _locate_index(self, index: int) -> FileIndexEntry:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        file_index = int(np.searchsorted(self._cumulative, index, side='right'))
        prev_cumulative = 0 if file_index == 0 else self._cumulative[file_index - 1]
        local_index = index - prev_cumulative
        return FileIndexEntry(file_index=file_index, local_index=local_index)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim == 0:
            array = array.reshape(1)
        return array.astype(np.float32, copy=False)

    def _prepare_array(self, logical_key: str, value: Any) -> np.ndarray:
        array = self._to_numpy(value)
        mode = str(self.reshape_map.get(logical_key, 'none')).lower()
        if mode == 'none':
            return array
        if mode == 'flatten':
            return array.reshape(-1)
        raise ValueError(f"Unsupported reshape mode '{mode}' for logical key '{logical_key}'")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entry = self._locate_index(index)
        handle = self._get_handle(entry.file_index)

        x = self._prepare_array('input', handle[self.key_map['input']][entry.local_index])
        sample: dict[str, torch.Tensor] = {
            'inputs': torch.as_tensor(x, dtype=self.input_dtype),
            'file_index': torch.tensor(entry.file_index, dtype=torch.long),
            'sample_index': torch.tensor(entry.local_index, dtype=torch.long),
        }

        if self.require_targets:
            bc_target = self._prepare_array('bc_target', handle[self.key_map['bc_target']][entry.local_index])
            vector_target = self._prepare_array('vector_target', handle[self.key_map['vector_target']][entry.local_index])
            reg_target = self._prepare_array('reg_target', handle[self.key_map['reg_target']][entry.local_index])
            sample['bc_target'] = torch.as_tensor(bc_target, dtype=self.target_dtype)
            sample['vector_target'] = torch.as_tensor(vector_target, dtype=self.target_dtype)
            sample['reg_target'] = torch.as_tensor(reg_target, dtype=self.target_dtype)

            sample_weight_key = self.key_map.get('sample_weight')
            if sample_weight_key and sample_weight_key in handle:
                sample_weight = self._prepare_array('sample_weight', handle[sample_weight_key][entry.local_index])
                sample['sample_weight'] = torch.as_tensor(sample_weight, dtype=self.target_dtype)

            bc_sample_weight_key = self.key_map.get('bc_sample_weight')
            if bc_sample_weight_key and bc_sample_weight_key in handle:
                bc_sample_weight = self._prepare_array('bc_sample_weight', handle[bc_sample_weight_key][entry.local_index])
                sample['bc_sample_weight'] = torch.as_tensor(bc_sample_weight, dtype=self.target_dtype)

            vector_sample_weight_key = self.key_map.get('vector_sample_weight')
            if vector_sample_weight_key and vector_sample_weight_key in handle:
                vector_sample_weight = self._prepare_array(
                    'vector_sample_weight',
                    handle[vector_sample_weight_key][entry.local_index],
                )
                sample['vector_sample_weight'] = torch.as_tensor(vector_sample_weight, dtype=self.target_dtype)

            reg_sample_weight_key = self.key_map.get('reg_sample_weight')
            if reg_sample_weight_key and reg_sample_weight_key in handle:
                reg_sample_weight = self._prepare_array('reg_sample_weight', handle[reg_sample_weight_key][entry.local_index])
                sample['reg_sample_weight'] = torch.as_tensor(reg_sample_weight, dtype=self.target_dtype)

        return sample


def build_key_map(config: Any) -> dict[str, str | None]:
    return {
        'input': config.data.keys.input,
        'bc_target': config.data.keys.bc_target,
        'vector_target': config.data.keys.vector_target,
        'reg_target': config.data.keys.reg_target,
        'sample_weight': config.data.keys.get('sample_weight'),
        'bc_sample_weight': config.data.keys.get('bc_sample_weight'),
        'vector_sample_weight': config.data.keys.get('vector_sample_weight'),
        'reg_sample_weight': config.data.keys.get('reg_sample_weight'),
    }


def build_datasets(config: Any) -> dict[str, MultiFileHDF5Dataset | None]:
    key_map = build_key_map(config)
    strict = bool(config.data.get_path('hdf5.strict', True))
    swmr = bool(config.data.get_path('hdf5.swmr', False))
    reshape_map = {
        'input': 'flatten',
        'vector_target': 'flatten',
    }
    user_reshape_map = config.data.get_path('preprocess.reshape')
    if isinstance(user_reshape_map, dict):
        for key, value in user_reshape_map.items():
            reshape_map[str(key)] = str(value)

    def _maybe_build(file_list: list[str]) -> MultiFileHDF5Dataset | None:
        if not file_list:
            return None
        return MultiFileHDF5Dataset(
            files=file_list,
            key_map=key_map,
            strict=strict,
            swmr=swmr,
            reshape_map=reshape_map,
            input_feature_dim=int(config.model.get('input_dim', 24)),
        )

    return {
        'train': _maybe_build(config.data.get('train_files', [])),
        'val': _maybe_build(config.data.get('val_files', [])),
        'test': _maybe_build(config.data.get('test_files', [])),
    }



def build_dataloader(dataset: Dataset | None, config: Any, train: bool) -> DataLoader | None:
    if dataset is None:
        return None
    loader_cfg = config.data.loader
    num_workers = int(loader_cfg.get('num_workers', 0))
    batch_size = int(loader_cfg.get('batch_size', 256))
    shuffle_train = bool(loader_cfg.get('shuffle_train', True))
    drop_last = bool(loader_cfg.get('drop_last', False)) if train else False

    common_kwargs: dict[str, Any] = {
        'num_workers': num_workers,
        'pin_memory': bool(loader_cfg.get('pin_memory', True)),
        'persistent_workers': bool(loader_cfg.get('persistent_workers', False)) if num_workers > 0 else False,
    }
    if num_workers > 0 and loader_cfg.get('prefetch_factor') is not None:
        common_kwargs['prefetch_factor'] = int(loader_cfg.get('prefetch_factor'))

    sampler_cfg = loader_cfg.get('sampler', {})
    sampler_name = 'default'
    if isinstance(sampler_cfg, dict):
        sampler_name = str(sampler_cfg.get('name', 'default')).lower()
    elif hasattr(sampler_cfg, 'get'):
        sampler_name = str(sampler_cfg.get('name', 'default')).lower()

    if train and sampler_name == 'contiguous_blocks':
        block_size_raw = sampler_cfg.get('block_size', max(batch_size * 8, 2048))
        batch_sampler = ContiguousBlockBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            block_size=int(block_size_raw),
            shuffle=shuffle_train,
            drop_last=drop_last,
            seed=int(config.get('seed', 42)),
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            **common_kwargs,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_train if train else False,
        drop_last=drop_last,
        **common_kwargs,
    )
