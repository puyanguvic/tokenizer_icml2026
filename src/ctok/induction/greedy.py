"""Greedy gain--distortion selection."""

from __future__ import annotations

from ctok.induction.distortion import DistortionEstimator, NullDistortion
from ctok.induction.gain import estimate_gains


def greedy_select(
    candidates: dict[str, int],
    budget: int,
    lambda_weight: float = 0.0,
    distortion: DistortionEstimator | None = None,
) -> list[str]:
    if budget <= 0:
        return []
    if distortion is None:
        distortion = NullDistortion()

    gains = estimate_gains(candidates)
    scored = []
    for token, gain in gains.items():
        score = gain - lambda_weight * distortion.score(token)
        scored.append((score, gain, token))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [token for _, _, token in scored[:budget]]
