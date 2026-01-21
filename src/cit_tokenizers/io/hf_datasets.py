from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class DatasetPullSpec:
    dataset: str
    split: str = "train"
    text_key: str = "text"
    label_key: Optional[str] = None
    subset: Optional[str] = None
    max_samples: Optional[int] = None


def pull_to_parquet(
    spec: DatasetPullSpec,
    out_path: str,
    *,
    shuffle: bool = False,
    seed: int = 0,
) -> Path:
    """Download a HuggingFace dataset split and save as a parquet file.

    Requires optional dependency: `datasets` (and `pyarrow` for parquet).
    This function is designed as a convenience helper for experiments.
    """

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "HuggingFace `datasets` is not installed. Install with: pip install datasets"
        ) from e

    ds = load_dataset(spec.dataset, spec.subset) if spec.subset else load_dataset(spec.dataset)
    if spec.split not in ds:
        raise KeyError(f"Split '{spec.split}' not found. Available: {list(ds.keys())}")

    split_ds = ds[spec.split]
    if shuffle:
        split_ds = split_ds.shuffle(seed=seed)
    if spec.max_samples is not None:
        split_ds = split_ds.select(range(int(spec.max_samples)))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    split_ds.to_parquet(str(out))
    return out
