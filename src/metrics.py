from __future__ import annotations

from typing import Sequence

from .dst.pipeline import TOKEN_SPLIT_RE
from .dst.tokenizer import DSTTokenizer


def _baseline_token_count(text: str) -> int:
    return len(TOKEN_SPLIT_RE.findall(text))


def round_trip_accuracy(tokenizer: DSTTokenizer, samples: Sequence[str]) -> float:
    """Fraction of samples that round-trip cleanly through encode/decode."""
    if not samples:
        return 1.0
    consistent = 0
    for sample in samples:
        encoded = tokenizer.encode(sample)
        decoded = tokenizer.decode(encoded)
        if decoded == tokenizer.normalizer(sample):
            consistent += 1
    return consistent / len(samples)


def compression_ratio(tokenizer: DSTTokenizer, samples: Sequence[str]) -> float:
    """
    Estimate compression ratio (tokens produced by DST vs. baseline split).
    Values < 1 indicate token savings.
    """
    baseline = 0
    dst_tokens = 0
    for sample in samples:
        normalized = tokenizer.normalizer(sample)
        baseline += _baseline_token_count(normalized)
        dst_tokens += len(tokenizer.encode_to_tokens(sample))
    if baseline == 0:
        return 0.0
    return dst_tokens / baseline


def average_token_length(tokenizer: DSTTokenizer) -> float:
    total_chars = 0
    total_tokens = 0
    for token in tokenizer.domain_tokens:
        total_chars += len(token)
        total_tokens += 1
    return total_chars / total_tokens if total_tokens else 0.0
