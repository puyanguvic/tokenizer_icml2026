"""Vocabulary induction utilities."""

from ctok_core.induction.candidates import collect_ngrams, collect_ngrams_with_labels
from ctok_core.induction.distortion import LabelEntropyDistortion, NullDistortion, build_label_entropy_distortion
from ctok_core.induction.greedy import greedy_select
from ctok_core.induction.gain import estimate_gains, gain_for

__all__ = [
    "collect_ngrams",
    "collect_ngrams_with_labels",
    "LabelEntropyDistortion",
    "NullDistortion",
    "build_label_entropy_distortion",
    "greedy_select",
    "estimate_gains",
    "gain_for",
]
