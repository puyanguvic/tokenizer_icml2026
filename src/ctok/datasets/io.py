"""Generic dataset loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ctok.datasets.config import DatasetConfig, DatasetExample


def _build_meta(payload: dict[str, Any], config: DatasetConfig) -> dict[str, Any] | None:
    if not config.group_field:
        return None
    if config.group_field in payload:
        return {config.group_field: payload[config.group_field]}
    return None


def _resolve_text(payload: dict[str, Any], config: DatasetConfig) -> str:
    if config.sequence_field and config.sequence_field in payload:
        sequence = payload[config.sequence_field]
        if isinstance(sequence, list):
            return config.sequence_delimiter.join(str(item) for item in sequence)
        return str(sequence)
    if config.text_fields:
        parts: list[str] = []
        for field in config.text_fields:
            value = payload.get(field)
            if value is None:
                continue
            parts.append(str(value))
        if parts:
            return config.field_delimiter.join(parts)
    if config.text_field in payload:
        return str(payload[config.text_field])
    available = ", ".join(sorted(payload.keys()))
    raise KeyError(f"Missing text field '{config.text_field}'. Available: {available}")


def _resolve_label(payload: dict[str, Any], config: DatasetConfig) -> str | int | None:
    if config.label_field is None:
        return None
    if config.label_field in payload:
        value = payload[config.label_field]
        return value if isinstance(value, (int, str)) else str(value)
    return None


def load_dataset(config: DatasetConfig) -> list[DatasetExample]:
    if config.hf_dataset:
        return _load_hf(config)
    if not config.path:
        raise ValueError("Dataset path is required for non-HF formats.")
    path = Path(config.path)
    if config.format == "text":
        return _load_text(path, config)
    if config.format == "jsonl":
        return _load_jsonl(path, config)
    if config.format in {"tsv", "csv"}:
        return _load_delimited(path, config)
    raise ValueError(f"Unsupported dataset format: {config.format}")


def _load_text(path: Path, config: DatasetConfig) -> list[DatasetExample]:
    examples: list[DatasetExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.rstrip("\n")
            if not text:
                continue
            examples.append(DatasetExample(text=text, label=None, meta=None))
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def _load_jsonl(path: Path, config: DatasetConfig) -> list[DatasetExample]:
    examples: list[DatasetExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            text = _resolve_text(payload, config)
            label = _resolve_label(payload, config)
            meta = _build_meta(payload, config)
            examples.append(DatasetExample(text=text, label=label, meta=meta))
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def _load_delimited(path: Path, config: DatasetConfig) -> list[DatasetExample]:
    examples: list[DatasetExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.rstrip("\n")
            if not line:
                continue
            if idx == 0 and config.has_header:
                continue
            fields = line.split(config.delimiter)
            if config.columns:
                payload = {name: value for name, value in zip(config.columns, fields)}
                text = _resolve_text(payload, config)
                label = _resolve_label(payload, config)
            else:
                if config.text_index >= len(fields):
                    raise ValueError(f"Missing text field in line {idx + 1}")
                text = fields[config.text_index]
                label = None
                if config.label_field is not None:
                    if config.label_index >= len(fields):
                        raise ValueError(f"Missing label field in line {idx + 1}")
                    label = fields[config.label_index]
            examples.append(DatasetExample(text=text, label=label, meta=None))
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def _load_hf(config: DatasetConfig) -> list[DatasetExample]:
    try:
        import datasets  # type: ignore
    except ImportError as exc:
        raise ImportError("Install the 'datasets' package to load HF datasets.") from exc

    if config.hf_streaming and config.max_samples is None:
        raise ValueError("Set max_samples when using streaming datasets.")

    dataset = datasets.load_dataset(
        config.hf_dataset,
        name=config.hf_name,
        split=config.hf_split,
        revision=config.hf_revision,
        streaming=config.hf_streaming,
    )

    examples: list[DatasetExample] = []
    for idx, row in enumerate(dataset):
        if config.max_samples is not None and idx >= config.max_samples:
            break
        payload = dict(row)
        text = _resolve_text(payload, config)
        label = _resolve_label(payload, config)
        meta = _build_meta(payload, config)
        examples.append(DatasetExample(text=text, label=label, meta=meta))

    if not examples:
        raise ValueError(f"No examples found in HF dataset {config.hf_dataset}")
    return examples
