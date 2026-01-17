"""Scheduling helpers for induction."""

from __future__ import annotations


def constant_schedule(value: float, steps: int) -> list[float]:
    return [value for _ in range(steps)]
