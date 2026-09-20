"""
lensllm.fallback
====================
Fallback function chain — tries providers in order on failure.

Usage in @track():
    @track(
        retries=3,
        fallback=[call_gemini, call_claude]
    )
    def call_llm(prompt: str) -> str:
        return call_openai(prompt)

    # If call_openai fails (even after retries):
    #   → tries call_gemini(prompt)
    # If call_gemini also fails:
    #   → tries call_claude(prompt)
    # If all fail:
    #   → raises the original exception from call_openai

Design:
    The fallback chain receives the same `prompt` as the primary function.
    Each fallback is called WITHOUT retries by default (the decorator handles
    retries on the primary). If you want retries on a fallback, wrap it with
    its own @track() decorator — they compose correctly because the decorator
    is provider-agnostic.

    All attempts (primary + fallbacks) are logged as separate spans sharing
    the same trace_id, so the dashboard shows the full fallback chain in one
    logical request.

    The final return value is the response from whichever function succeeded.
    If none succeed, the last exception is re-raised.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class FallbackResult:
    """
    Result from a fallback chain execution.

    Attributes:
        response:        The successful response (None if all failed).
        succeeded_with:  Name of the function that returned the response.
        attempts:        List of (function_name, error_or_None) in order tried.
        total_latency_ms: Wall-clock time for the entire chain.
    """
    response: Any
    succeeded_with: Optional[str]
    attempts: list[tuple[str, Optional[str]]] = field(default_factory=list)
    total_latency_ms: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.succeeded_with is not None


class FallbackChain:
    """
    Tries a sequence of callable providers in order, returning the first
    successful response.

    Args:
        providers: Ordered list of callables (primary is handled by the decorator;
                   this chain handles the fallbacks only).
    """

    def __init__(self, providers: list[Callable]):
        if not providers:
            raise ValueError("FallbackChain requires at least one provider function.")
        self._providers = providers

    def run(self, prompt: str, *args: Any, **kwargs: Any) -> FallbackResult:
        """
        Try each provider in order until one succeeds.

        Args:
            prompt: The prompt string to pass to each provider.
            *args:  Additional positional args forwarded to each provider.
            **kwargs: Keyword args forwarded to each provider.

        Returns:
            FallbackResult with the successful response and audit trail.
        """
        start_total = time.perf_counter()
        attempts: list[tuple[str, Optional[str]]] = []

        for provider in self._providers:
            name = getattr(provider, "__name__", repr(provider))
            try:
                response = provider(prompt, *args, **kwargs)
                total_ms = (time.perf_counter() - start_total) * 1000
                attempts.append((name, None))  # None = no error
                return FallbackResult(
                    response=response,
                    succeeded_with=name,
                    attempts=attempts,
                    total_latency_ms=round(total_ms, 2),
                )
            except Exception as exc:
                attempts.append((name, str(exc)))
                # Continue to next provider

        total_ms = (time.perf_counter() - start_total) * 1000
        return FallbackResult(
            response=None,
            succeeded_with=None,
            attempts=attempts,
            total_latency_ms=round(total_ms, 2),
        )
