"""
Domain-Specific Tokenization (DST) core package.

This module exposes the high-level builder that orchestrates vocabulary
induction, DFST compilation, and serialization as described in the paper.
"""

from .pipeline import build_dst_tokenizer, CandidateExtractorConfig, ScoreWeights
from .tokenizer import DSTTokenizer
from .signals import GradientSignalProvider

__all__ = [
    "build_dst_tokenizer",
    "CandidateExtractorConfig",
    "ScoreWeights",
    "DSTTokenizer",
    "GradientSignalProvider",
]
