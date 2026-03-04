from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


class ConfigurationError(ValueError):
    """Raised when the YAML configuration is invalid."""


class ConfigNode(dict):
    """Recursive dict with attribute-style access."""

    def __init__(self, mapping: dict[str, Any] | None = None):
        super().__init__()
        mapping = mapping or {}
        for key, value in mapping.items():
            self[key] = self._convert(value)

    @classmethod
    def _convert(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._convert(item) for item in value]
        return value

    def __getattribute__(self, item: str) -> Any:
        if not item.startswith('_'):
            try:
                return dict.__getitem__(self, item)
            except KeyError:
                pass
        return super().__getattribute__(item)

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._convert(value)

    def clone(self) -> "ConfigNode":
        return ConfigNode(copy.deepcopy(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, ConfigNode):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, ConfigNode) else item for item in value]
            else:
                result[key] = value
        return result

    def get_path(self, dotted_path: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted_path.split('.'):
            if isinstance(node, ConfigNode) and part in node:
                node = node[part]
            else:
                return default
        return node


def _expand_path(path_like: str, base_dir: Path) -> str:
    raw = os.path.expandvars(os.path.expanduser(path_like))
    path = Path(raw)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _resolve_path_list(paths: Iterable[str] | None, base_dir: Path) -> list[str]:
    if not paths:
        return []
    return [_expand_path(path, base_dir) for path in paths]


def _validate_section(config: ConfigNode, dotted_path: str) -> None:
    if config.get_path(dotted_path) is None:
        raise ConfigurationError(f"Missing required configuration section: '{dotted_path}'")


def validate_config(config: ConfigNode) -> None:
    required_paths = [
        'data',
        'data.keys',
        'model',
        'model.trunk',
        'model.heads.bc',
        'model.heads.vector_regression',
        'model.heads.regression',
        'losses.bc',
        'losses.vector_regression',
        'losses.regression',
        'multitask.loss_balancer',
        'multitask.gradient_surgery',
        'training.optimizer',
    ]
    for path in required_paths:
        _validate_section(config, path)

    architecture = str(config.model.get('architecture', 'mlp')).lower()
    if architecture not in {'mlp', 'equivariant_basis'}:
        raise ConfigurationError(
            f"Unsupported model.architecture '{architecture}'. Choose from mlp or equivariant_basis."
        )

    input_dim = config.get_path('model.input_dim')
    if input_dim != 24:
        raise ConfigurationError(f"model.input_dim must be 24, got {input_dim!r}")

    train_files = config.get_path('data.train_files', [])
    val_files = config.get_path('data.val_files', [])
    test_files = config.get_path('data.test_files', [])
    if not train_files and not val_files and not test_files:
        raise ConfigurationError('At least one of data.train_files, data.val_files, or data.test_files must be provided.')

    balancer_name = str(config.multitask.loss_balancer.name).lower()
    if balancer_name not in {'static', 'kendall_gal', 'gradnorm'}:
        raise ConfigurationError(
            f"Unsupported loss balancer '{balancer_name}'. Choose from static, kendall_gal, gradnorm."
        )

    surgery_name = str(config.multitask.gradient_surgery.name).lower()
    if surgery_name not in {'none', 'pcgrad'}:
        raise ConfigurationError(
            f"Unsupported gradient_surgery '{surgery_name}'. Choose from none or pcgrad."
        )

    if surgery_name == 'pcgrad' and balancer_name != 'static':
        raise ConfigurationError(
            'PCGrad is configured as gradient surgery. In this starter pipeline it is supported with '
            'multitask.loss_balancer.name=static to keep optimizer semantics sane.'
        )

    accumulation_steps = int(config.training.get('gradient_accumulation_steps', 1))
    if accumulation_steps < 1:
        raise ConfigurationError('training.gradient_accumulation_steps must be >= 1')
    if accumulation_steps != 1:
        raise ConfigurationError(
            'This starter pipeline currently expects training.gradient_accumulation_steps=1. '
            'That field is included as a placeholder so you can extend it cleanly later.'
        )

    if config.model.heads.bc.output_dim < 1:
        raise ConfigurationError('model.heads.bc.output_dim must be >= 1')
    if config.model.heads.vector_regression.output_dim < 1:
        raise ConfigurationError('model.heads.vector_regression.output_dim must be >= 1')
    if config.model.heads.regression.output_dim != 1:
        raise ConfigurationError('model.heads.regression.output_dim must be exactly 1 for scalar regression.')



def load_config(config_path: str | os.PathLike[str]) -> ConfigNode:
    path = Path(config_path).expanduser().resolve()
    with path.open('r', encoding='utf-8') as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raise ConfigurationError(f'Configuration file {path} is empty.')

    config = ConfigNode(raw)
    config._config_path = str(path)
    config._config_dir = str(path.parent)

    base_dir = path.parent
    if 'data' in config:
        config.data.train_files = _resolve_path_list(config.data.get('train_files'), base_dir)
        config.data.val_files = _resolve_path_list(config.data.get('val_files'), base_dir)
        config.data.test_files = _resolve_path_list(config.data.get('test_files'), base_dir)

    if 'output' in config and 'dir' in config.output:
        config.output.dir = _expand_path(config.output.dir, base_dir)

    validate_config(config)
    return config
