"""Tokenization runtime package."""

from ctok_core.tokenization.boundary import DEFAULT_BOUNDARY_CHARS, normalize_boundary_chars
from ctok_core.tokenization.runtime import BoundaryAwareTokenizerRuntime, TokenizerRuntime
from ctok_core.tokenization.rules import RuleSet
from ctok_core.tokenization.tokenizer import CtokTokenizer
from ctok_core.tokenization.vocab import Vocabulary

try:  # pragma: no cover - optional dependency
    from ctok_core.tokenization.hf import CtokHFTokenizer
except ImportError:  # pragma: no cover - optional dependency

    class _MissingCtokHFTokenizer:  # noqa: D401 - small shim
        """Placeholder that raises if transformers is missing."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise ImportError("Install 'transformers' to use CtokHFTokenizer.")

    CtokHFTokenizer = _MissingCtokHFTokenizer  # type: ignore[assignment]

__all__ = [
    "TokenizerRuntime",
    "BoundaryAwareTokenizerRuntime",
    "RuleSet",
    "Vocabulary",
    "CtokTokenizer",
    "CtokHFTokenizer",
    "DEFAULT_BOUNDARY_CHARS",
    "normalize_boundary_chars",
]
