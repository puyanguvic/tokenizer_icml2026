from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pyarrow as pa


@dataclass(frozen=True)
class PreprocessSpec:
    name: str
    required_columns: List[str]
    text_key: str = "text"
    label_key: Optional[str] = "label"


PreprocessFn = Callable[[pa.RecordBatch], pa.RecordBatch]

_REGISTRY: Dict[str, Tuple[PreprocessSpec, PreprocessFn]] = {}


def register(spec: PreprocessSpec):
    def deco(fn: PreprocessFn):
        _REGISTRY[spec.name] = (spec, fn)
        return fn

    return deco


def get(name: str) -> Tuple[PreprocessSpec, PreprocessFn]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown preprocess '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def available() -> List[str]:
    return sorted(_REGISTRY.keys())
