"""Dataset configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetExample:
    text: str
    label: str | int | None = None
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: str = ""
    split: str = "train"
    task: str = "classification"
    input_type: str = "text"
    format: str = "text"
    text_field: str = "text"
    text_fields: list[str] | None = None
    label_field: str | None = "label"
    label_mapping: dict[str, int] | None = None
    sequence_field: str | None = None
    group_field: str | None = None
    delimiter: str = "\t"
    columns: list[str] | None = None
    text_index: int = 1
    label_index: int = 0
    has_header: bool = False
    sequence_delimiter: str = "\n"
    field_delimiter: str = "\n"
    hf_dataset: str | None = None
    hf_name: str | None = None
    hf_split: str = "train"
    hf_revision: str | None = None
    hf_streaming: bool = False
    max_samples: int | None = None
