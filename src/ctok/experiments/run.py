"""Experiment runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ctok.datasets.config import DatasetConfig
from ctok.models.train import train_roberta


def run_experiment(
    dataset_config_path: str | Path,
    model_config_path: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    dataset_config = _load_dataset_config(Path(dataset_config_path))
    model_name = _load_model_name(Path(model_config_path))
    return train_roberta(
        dataset_config=dataset_config,
        model_name=model_name,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        **kwargs,
    )


def _load_dataset_config(path: Path) -> DatasetConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Dataset config must be a mapping.")
    return DatasetConfig(**payload)


def _load_model_name(path: Path) -> str:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Model config must be a mapping.")
    hf_id = payload.get("hf_id")
    if not hf_id:
        raise ValueError("Model config must include 'hf_id'.")
    return str(hf_id)
