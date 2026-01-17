"""Distortion estimation interfaces."""

from __future__ import annotations

import math
from typing import Iterable


class DistortionEstimator:
    def score(self, token: str) -> float:
        raise NotImplementedError


class NullDistortion(DistortionEstimator):
    def score(self, token: str) -> float:
        return 0.0


class LabelEntropyDistortion(DistortionEstimator):
    """Proxy distortion based on label entropy of token occurrences."""

    def __init__(self, label_counts: dict[str, dict[str, int]], smoothing: float = 1e-6) -> None:
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self._label_counts = label_counts
        labels = set()
        for counts in label_counts.values():
            labels.update(counts.keys())
        self._labels = sorted(labels)
        self._smoothing = smoothing

    def score(self, token: str) -> float:
        counts = self._label_counts.get(token)
        if not counts:
            return 0.0
        total = sum(counts.values())
        if total == 0:
            return 0.0
        num_labels = max(len(self._labels), 1)
        denom = total + self._smoothing * num_labels
        entropy = 0.0
        for label in self._labels:
            value = counts.get(label, 0)
            prob = (value + self._smoothing) / denom
            entropy -= prob * math.log(prob)
        return entropy * total


def build_label_entropy_distortion(
    label_counts: dict[str, dict[str, int]],
    smoothing: float = 1e-6,
) -> LabelEntropyDistortion:
    return LabelEntropyDistortion(label_counts=label_counts, smoothing=smoothing)
