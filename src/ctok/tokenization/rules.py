"""Rule representation for deterministic tokenization."""

from __future__ import annotations

from dataclasses import dataclass

from ctok.tokenization.vocab import Vocabulary


@dataclass(frozen=True)
class RuleSet:
    tokens: list[str]
    special_tokens: dict[str, str]
    version: int = 1

    @classmethod
    def from_vocab(cls, vocab: Vocabulary) -> "RuleSet":
        return cls(tokens=list(vocab.tokens), special_tokens=dict(vocab.special_tokens))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "tokens": self.tokens,
            "special_tokens": self.special_tokens,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RuleSet":
        return cls(
            tokens=list(payload.get("tokens", [])),
            special_tokens=dict(payload.get("special_tokens", {})),
            version=int(payload.get("version", 1)),
        )
