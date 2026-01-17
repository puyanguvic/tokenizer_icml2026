"""Gain estimation for candidate tokens."""

from __future__ import annotations


def gain_for(token: str, freq: int) -> float:
    return max(len(token) - 1, 0) * float(freq)


def estimate_gains(candidates: dict[str, int]) -> dict[str, float]:
    return {token: gain_for(token, freq) for token, freq in candidates.items()}
