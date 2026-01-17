"""Simple dataset registry."""

from __future__ import annotations

from collections.abc import Callable

from ctok_extras.datasets.config import DatasetConfig, DatasetExample
from ctok_extras.datasets.io import load_dataset

DatasetLoader = Callable[[DatasetConfig], list[DatasetExample]]

_REGISTRY: dict[str, DatasetLoader] = {}


def register(name: str) -> Callable[[DatasetLoader], DatasetLoader]:
    def decorator(func: DatasetLoader) -> DatasetLoader:
        _REGISTRY[name] = func
        return func

    return decorator


def get(name: str) -> DatasetLoader:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset: {name}")
    return _REGISTRY[name]


def load(config: DatasetConfig) -> list[DatasetExample]:
    loader = _REGISTRY.get(config.name)
    if loader is None:
        return load_dataset(config)
    return loader(config)


def list_datasets() -> list[str]:
    return sorted(_REGISTRY.keys())
