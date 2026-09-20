"""
gateway_sdk.retry
=================
Exponential backoff with jitter — all parameters are developer-adjustable.

Algorithm:
    For attempt i in [1, max_retries]:
        1. Call fn()
        2. On success → return result immediately
        3. On a retryable exception:
            a. Compute delay = min(base_delay * 2^(i-1), max_delay)
            b. If jitter=True: delay = delay * uniform(0.5, 1.5)
            c. sleep(delay) and retry
        4. After max_retries exhausted → re-raise the last exception

Why exponential backoff + jitter?
    Exponential backoff:
        Doubles the wait on each retry (1s → 2s → 4s → 8s…). This gives
        the provider time to recover without hammering it at a fixed rate.
        `max_delay` caps the wait so it never grows unbounded.

    Jitter:
        Without jitter, if 100 clients all hit a 429 rate-limit at the same
        time, they all wait exactly 2s and then ALL retry simultaneously —
        causing another thundering herd. Jitter spreads retries randomly
        across a window, smoothing the load on the provider.

        We use "full jitter": delay = random(0, computed_delay).
        This is the approach recommended by the AWS Architecture Blog:
        https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable, Tuple, Type


# Default exception types that should trigger a retry.
# These cover transient network errors and common LLM provider HTTP errors.
DEFAULT_RETRY_ON: Tuple[Type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def retry_with_backoff(
    fn: Callable,
    args: tuple = (),
    kwargs: dict | None = None,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retry_on: Tuple[Type[Exception], ...] = DEFAULT_RETRY_ON,
) -> Any:
    """
    Call `fn(*args, **kwargs)` with exponential backoff retry on failure.

    All parameters are developer-adjustable via @track():
        max_retries  — total retry attempts after the first failure (default 3)
        base_delay   — initial wait in seconds (default 1.0s)
        max_delay    — maximum wait cap in seconds (default 60s)
        jitter       — randomise delay to prevent thundering herd (default True)
        retry_on     — tuple of Exception types to retry on (default: network errors)

    Args:
        fn:          The callable to invoke (the wrapped LLM function).
        args:        Positional arguments to pass to fn.
        kwargs:      Keyword arguments to pass to fn.
        max_retries: How many times to retry after the first failure.
        base_delay:  Base wait time for the first retry (seconds).
        max_delay:   Maximum wait time cap (seconds).
        jitter:      If True, apply full jitter to the computed delay.
        retry_on:    Tuple of exception types that trigger a retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception from fn() if all retries are exhausted.
        Any exception NOT in `retry_on` is re-raised immediately.
    """
    if kwargs is None:
        kwargs = {}

    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):  # attempt 0 = first try
        try:
            return fn(*args, **kwargs)

        except retry_on as exc:
            last_exc = exc

            if attempt == max_retries:
                # All retries exhausted — raise the last exception
                raise

            # Compute exponential delay: base * 2^attempt
            delay = min(base_delay * (2 ** attempt), max_delay)

            if jitter:
                # Full jitter: uniformly random between 0 and the computed delay.
                # This is statistically better than half-jitter for reducing
                # collision probability under high concurrency.
                delay = random.uniform(0, delay)

            time.sleep(delay)

        except Exception:
            # Non-retryable exception — propagate immediately without retry
            raise

    # Should never reach here (loop always raises or returns), but satisfies type checkers
    assert last_exc is not None
    raise last_exc


def compute_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    jitter: bool,
) -> float:
    """
    Compute the retry delay for a given attempt number.

    Exposed as a standalone function so it can be tested in isolation
    without actually sleeping.

    Args:
        attempt:    0-indexed retry number (0 = first retry after initial failure).
        base_delay: Base delay in seconds.
        max_delay:  Maximum delay cap in seconds.
        jitter:     Whether to apply full jitter.

    Returns:
        The computed delay in seconds (float).
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    if jitter:
        delay = random.uniform(0, delay)
    return delay
