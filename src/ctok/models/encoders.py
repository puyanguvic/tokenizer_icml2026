"""Encoder adapters."""

from __future__ import annotations

from typing import Any


def load_roberta_classifier(
    hf_id: str,
    num_labels: int,
    id2label: dict[int, str] | None = None,
    label2id: dict[str, int] | None = None,
    **kwargs: Any,
) -> Any:
    try:
        from transformers import AutoConfig, AutoModelForSequenceClassification
    except ImportError as exc:
        raise ImportError("Install 'transformers' to load RoBERTa models.") from exc

    config = AutoConfig.from_pretrained(
        hf_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        **kwargs,
    )
    return AutoModelForSequenceClassification.from_pretrained(hf_id, config=config)


def load_encoder(*args: Any, **kwargs: Any) -> Any:
    return load_roberta_classifier(*args, **kwargs)
