from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .runtime import CITArtifact

@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str

def validate_typed_symbol_integrity(vocab_tokens: Iterable[str], typed_symbols: Sequence[str]) -> list[ValidationIssue]:
    """Validate that vocab does not contain proper substrings of typed symbols.

    This enforces the paper's 'typed-symbol integrity' constraint at the artifact level.
    """
    vocab = set(vocab_tokens)
    issues: list[ValidationIssue] = []
    for sym in typed_symbols:
        for tok in vocab:
            if tok == sym:
                continue
            if tok and tok in sym:
                issues.append(
                    ValidationIssue(
                        code="typed_symbol_substring",
                        message=f"Vocab token '{tok}' is a proper substring of typed symbol '{sym}'.",
                    )
                )
                # don't spam too much
                break
    return issues

def validate_artifact(art: CITArtifact, typed_symbols: Sequence[str]) -> list[ValidationIssue]:
    issues = []
    issues.extend(validate_typed_symbol_integrity(art.vocab.keys(), typed_symbols))
    return issues
