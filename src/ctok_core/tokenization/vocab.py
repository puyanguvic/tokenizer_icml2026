"""Vocabulary utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Vocabulary:
    tokens: list[str]
    special_tokens: dict[str, str]

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("Vocabulary is empty.")
        seen = set()
        for token in self.tokens:
            if not token:
                raise ValueError("Empty token in vocabulary.")
            if token in seen:
                raise ValueError(f"Duplicate token: {token}")
            seen.add(token)
        for name, token in self.special_tokens.items():
            if token not in seen:
                raise ValueError(f"Special token '{name}' missing from tokens.")

    def token_to_id(self) -> dict[str, int]:
        return {token: idx for idx, token in enumerate(self.tokens)}

    def id_for(self, token: str) -> int:
        mapping = self.token_to_id()
        if token not in mapping:
            raise KeyError(f"Token not in vocabulary: {token}")
        return mapping[token]

    def token_for(self, idx: int) -> str:
        return self.tokens[idx]

    def special_ids(self) -> dict[str, int]:
        mapping = self.token_to_id()
        return {name: mapping[token] for name, token in self.special_tokens.items()}

    def with_special_tokens(self, special_tokens: dict[str, str]) -> "Vocabulary":
        tokens = list(self.tokens)
        merged = dict(self.special_tokens)
        for name, token in special_tokens.items():
            if token not in tokens:
                tokens.append(token)
            merged[name] = token
        return Vocabulary(tokens=tokens, special_tokens=merged)

    def to_dict(self) -> dict[str, object]:
        return {
            "tokens": self.tokens,
            "special_tokens": self.special_tokens,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Vocabulary":
        tokens = list(payload.get("tokens", []))
        special_tokens = dict(payload.get("special_tokens", {}))
        return cls(tokens=tokens, special_tokens=special_tokens)
