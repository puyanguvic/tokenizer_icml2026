"""Boundary utilities for tokenizer stability."""

from __future__ import annotations

from collections.abc import Iterable

DEFAULT_BOUNDARY_CHARS = set(" \t\n\r/?:=&;@#,.<>[]{}()\"'|")


def normalize_boundary_chars(value: str | Iterable[str] | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return set(value)
    result: set[str] = set()
    for item in value:
        result.update(item)
    return result
