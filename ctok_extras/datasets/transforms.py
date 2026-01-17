"""Normalization transforms."""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def canonicalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")
