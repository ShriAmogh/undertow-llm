"""
lensllm.canary
==================
Canary / A-B traffic routing for @track().

Allows a fraction of LLM calls to be routed to an alternative (canary)
function — for example, a different model or provider — while the rest
go to the primary function.

Both variants are executed and logged with variant='primary' or
variant='canary' so the dashboard can compare latency, cost, and error rate.

Usage:
    @track(
        canary={
            "fn": call_gpt4,      # the canary function (same signature)
            "weight": 0.10,       # route 10% of cache-miss traffic here
        }
    )
    def call_gemini(prompt: str) -> str:
        ...

Design note — determinism:
    random.random() is used for routing. This is intentionally non-deterministic
    (same prompt may route differently on consecutive calls), which is the
    correct behaviour for true traffic sampling. If you need per-user or
    per-session determinism, pass a seed externally.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class CanaryConfig:
    """Validated canary routing configuration."""
    fn: Callable
    weight: float      # 0.0 = never route to canary; 1.0 = always route


def parse_canary(canary: dict | None) -> Optional[CanaryConfig]:
    """
    Parse the raw canary dict passed to @track() into a CanaryConfig.

    Returns None if canary is None or empty.

    Raises:
        ValueError — on invalid weight or missing 'fn' key.
    """
    if not canary:
        return None

    fn = canary.get("fn")
    if fn is None or not callable(fn):
        raise ValueError(
            "canary dict must contain 'fn': a callable with the same signature "
            "as the primary function."
        )

    weight = canary.get("weight", 0.0)
    if not (0.0 <= weight <= 1.0):
        raise ValueError(
            f"canary 'weight' must be between 0.0 and 1.0, got {weight!r}"
        )

    return CanaryConfig(fn=fn, weight=weight)


def should_use_canary(weight: float) -> bool:
    """
    Return True with probability = weight.

    Args:
        weight: Float in [0.0, 1.0]
    """
    if weight <= 0.0:
        return False
    if weight >= 1.0:
        return True
    return random.random() < weight
