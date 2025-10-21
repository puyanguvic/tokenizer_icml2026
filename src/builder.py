from __future__ import annotations

import dataclasses
import json
import os
from typing import Callable, Dict, List, Optional, Sequence

try:
    from .dst import CandidateExtractorConfig, ScoreWeights, build_dst_tokenizer
    from .http_tokenizer import HTTP_GRAMMAR_PATTERNS, http_clean_line
except ImportError:  # pragma: no cover - fallback for script usage
    from dst import CandidateExtractorConfig, ScoreWeights, build_dst_tokenizer
    from http_tokenizer import HTTP_GRAMMAR_PATTERNS, http_clean_line


def _iter_dataset_records(dataset, text_field: str) -> Sequence[str]:
    """
    Normalize arbitrary dataset-like inputs into a flat sequence of text strings.

    Supported inputs:
      - Hugging Face `datasets.Dataset` or `DatasetDict`
      - Iterable of dicts/lists with a specified `text_field`
      - Iterable of raw strings
    """

    def _extract_text(example):
        if isinstance(example, dict):
            if text_field not in example:
                raise KeyError(f"Field '{text_field}' not present in dataset example.")
            value = example[text_field]
        elif isinstance(example, (list, tuple)):
            if not example:
                return ""
            # Assume field index 0 if using list-style dataset.
            value = example[0]
        else:
            value = example
        return "" if value is None else str(value)

    try:
        from datasets import Dataset, DatasetDict  # type: ignore
    except ImportError:
        Dataset = DatasetDict = None  # type: ignore

    records: List[str] = []

    if DatasetDict is not None and isinstance(dataset, DatasetDict):
        for split in dataset.values():
            records.extend(_extract_text(example) for example in split)
        return records

    if Dataset is not None and isinstance(dataset, Dataset):
        records.extend(_extract_text(example) for example in dataset)
        return records

    if isinstance(dataset, dict):
        for value in dataset.values():
            records.extend(_iter_dataset_records(value, text_field))
        return records

    if isinstance(dataset, (list, tuple, set)):
        records.extend(_extract_text(example) for example in dataset)
        return records

    if hasattr(dataset, "__iter__"):
        for example in dataset:
            records.append(_extract_text(example))
        return records

    raise TypeError(
        "Unsupported dataset type. Provide a Hugging Face Dataset, DatasetDict, or an iterable of records."
    )


def train_domain_tokenizer(
    dataset,
    *,
    text_field: str = "text",
    save_dir: str = "tokenizer_output",
    vocab_size: int | None = 32000,
    normalizer: Callable[[str], str] = http_clean_line,
    config: CandidateExtractorConfig | None = None,
    config_overrides: Optional[Dict[str, object]] = None,
    gradient_provider: Optional[Callable[[Sequence[str], Sequence[str]], Dict[str, float]]] = None,
    gradient_scores: Optional[Dict[str, float]] = None,
    mutual_info_scores: Optional[Dict[str, float]] = None,
) -> None:
    """
    Build a Domain-Specific Tokenizer following the workflow in the DST paper.

    Parameters
    ----------
    dataset:
        Dataset supplying raw text (Hugging Face Dataset/DatasetDict or iterable).
    text_field:
        Name of the field/column containing text strings.
    save_dir:
        Destination directory for serialized artifacts.
    vocab_size:
        Maximum vocabulary size (including special + fallback tokens).
    normalizer:
        Pre-processing function applied before tokenization.
    config:
        Optional pre-constructed CandidateExtractorConfig.
    config_overrides:
        Dict of fields that override the config (e.g., {"min_frequency": 10}).
    gradient_provider:
        Callable used to compute gradient salience per candidate.
    gradient_scores:
        Precomputed gradient scores (used if provider is not supplied).
    mutual_info_scores:
        Optional overrides for mutual-information scores.
    """
    os.makedirs(save_dir, exist_ok=True)
    corpus_lines = _iter_dataset_records(dataset, text_field=text_field)

    if config is None:
        config = CandidateExtractorConfig(
            max_vocab=vocab_size or 32000,
            grammar_patterns=HTTP_GRAMMAR_PATTERNS,
        )
    elif vocab_size is not None:
        config = dataclasses.replace(config, max_vocab=vocab_size)

    if config_overrides:
        overrides = dict(config_overrides)
        if "weights" in overrides and isinstance(overrides["weights"], dict):
            overrides["weights"] = ScoreWeights(**overrides["weights"])
        config = dataclasses.replace(config, **overrides)

    tokenizer = build_dst_tokenizer(
        corpus=corpus_lines,
        normalizer=normalizer,
        config=config,
        save_dir=save_dir,
        gradient_provider=gradient_provider,
        gradient_scores=gradient_scores,
        mutual_info_scores=mutual_info_scores,
    )

    metadata = {
        "max_vocab": config.max_vocab,
        "min_frequency": config.min_frequency,
        "min_length": config.min_length,
        "max_tokens": config.max_tokens,
        "special_tokens": list(config.special_tokens),
        "grammar_patterns": list(config.grammar_patterns),
        "preserve_case": config.preserve_case,
        "scoring_weights": dataclasses.asdict(config.weights),
        "num_domain_tokens": len(tokenizer.domain_tokens),
        "num_fallback_tokens": len(tokenizer.fallback_tokens),
    }
    with open(
        os.path.join(save_dir, "tokenizer_metadata.json"), "w", encoding="utf-8"
    ) as fp:
        json.dump(metadata, fp, indent=2)

    print(f"✅ DST tokenizer exported to {save_dir}")
