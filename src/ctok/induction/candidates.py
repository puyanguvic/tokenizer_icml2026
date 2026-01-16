"""Candidate token generation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable


def collect_ngrams(
    lines: list[str],
    min_len: int = 2,
    max_len: int = 8,
    min_freq: int = 2,
    boundary_chars: set[str] | None = None,
) -> dict[str, int]:
    if min_len < 1 or max_len < min_len:
        raise ValueError("Invalid ngram lengths.")

    counts: Counter[str] = Counter()
    for line in lines:
        for span in _iter_spans(line, boundary_chars):
            length = len(span)
            for n in range(min_len, min(max_len, length) + 1):
                for i in range(0, length - n + 1):
                    counts[span[i : i + n]] += 1

    return {token: freq for token, freq in counts.items() if freq >= min_freq}


def collect_ngrams_with_labels(
    lines: list[str],
    labels: Iterable[str | int],
    min_len: int = 2,
    max_len: int = 8,
    min_freq: int = 2,
    boundary_chars: set[str] | None = None,
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    if min_len < 1 or max_len < min_len:
        raise ValueError("Invalid ngram lengths.")

    labels_list = list(labels)
    if len(labels_list) != len(lines):
        raise ValueError("labels must match the number of lines")

    counts: Counter[str] = Counter()
    label_counts: dict[str, Counter[str]] = {}

    for line, label in zip(lines, labels_list):
        label_str = str(label)
        for span in _iter_spans(line, boundary_chars):
            length = len(span)
            for n in range(min_len, min(max_len, length) + 1):
                for i in range(0, length - n + 1):
                    token = span[i : i + n]
                    counts[token] += 1
                    counter = label_counts.setdefault(token, Counter())
                    counter[label_str] += 1

    filtered_counts = {token: freq for token, freq in counts.items() if freq >= min_freq}
    filtered_labels: dict[str, dict[str, int]] = {}
    for token, freq in filtered_counts.items():
        filtered_labels[token] = dict(label_counts.get(token, {}))

    return filtered_counts, filtered_labels


def _iter_spans(line: str, boundary_chars: set[str] | None) -> list[str]:
    if not boundary_chars:
        return [line]
    spans: list[str] = []
    start = 0
    for idx, ch in enumerate(line):
        if ch not in boundary_chars:
            continue
        if start < idx:
            spans.append(line[start:idx])
        start = idx + 1
    if start < len(line):
        spans.append(line[start:])
    return spans
