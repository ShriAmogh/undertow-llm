"""
lensllm.decorator
=====================
Core @track() decorator — Phases 2-10 implementation.

Call flow on every wrapped function invocation:
    1.  Ensure DB is initialised (idempotent, fast after first call)
    2.  Resolve trace/span IDs (Phase 7: inherit from parent if nested)
    3.  Safety policy check (Phase 8: if policy callable provided)
    4.  Embed the prompt
    5.  If cache=True → check cache
        a. HIT  → log (hit, latency=0) → return cached response immediately
        b. MISS → fall through
    6.  Apply rate limiter (Phase 4)
    7.  Select primary vs canary function (Phase 9)
    8.  Call the wrapped function with retry (Phase 3)
        a. On success → go to step 9
        b. On all retries exhausted → try fallback chain (Phase 3)
        c. If fallback also fails → log error, re-raise
    9. If response is a generator → wrap for streaming (Phase 5)
        (logging + cache write happen inside the stream wrapper on exhaustion)
    10. Log the request (miss, latency, response, variant, cost_per_call)
    11. Write response to cache
    12. Return response to caller, completely unmodified

Parameters:
    cache, similarity_threshold, cache_ttl,
    retries, base_delay, max_delay, jitter, retry_on, fallback,
    rate_limit_tokens, rate_limit_rate, max_concurrency,
    cost_per_call, policy, canary
"""

from __future__ import annotations

import functools
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Type

from lensllm.config import get_config
from lensllm.db import init_db
from lensllm.cache.semantic import embed
from lensllm.backends.factory import get_cache_backend, get_metrics_store
from lensllm.logging.store import LogEntry
from lensllm.retry import retry_with_backoff, DEFAULT_RETRY_ON
from lensllm.fallback import FallbackChain
from lensllm.rate_limiter import get_limiter
from lensllm.queue import get_queue
from lensllm.streaming import is_generator, wrap_stream
from lensllm.tracing import get_current_trace, set_current_trace, reset_current_trace
from lensllm.canary import parse_canary, should_use_canary


class PolicyViolationError(Exception):
    """Raised when a policy hook returns 'block'."""


@dataclass
class CallInfo:
    """Metadata about the most recent @track call in this context."""
    cache_hit: bool       # True if response came from cache
    latency_ms: float     # Wall-clock time of the full call (incl. embed)
    fn_name: str          # Name of the wrapped function


# Per-context (thread / async task) storage for last call metadata.
# Callers read this via get_last_call_info() immediately after their call.
_last_call_info: ContextVar[Optional[CallInfo]] = ContextVar(
    "_last_call_info", default=None
)


def get_last_call_info() -> Optional[CallInfo]:
    """
    Return metadata about the most recent @track-wrapped call made in this
    context (thread or async task).

    Returns None if no @track call has been made yet.

    Example::

        resp = ask_gemini(prompt)
        info = get_last_call_info()
        label = "⚡ cached" if info and info.cache_hit else "🌐 live"
    """
    return _last_call_info.get()


def track(
    # ── Cache ─────────────────────────────────────────────────────────────────
    cache: bool | None = None,
    similarity_threshold: float | None = None,
    cache_ttl: int | None = None,          # seconds; None = no expiry

    # ── Retry ─────────────────────────────────────────────────────────────────
    retries: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    jitter: bool | None = None,
    retry_on: Tuple[Type[Exception], ...] | None = None,
    fallback: list[Callable] | None = None,

    # ── Rate Limiting & Concurrency ───────────────────────────────────────────
    rate_limit_tokens: float | None = None,
    rate_limit_rate: float | None = None,
    max_concurrency: int | None = None,

    # ── Cost Override ─────────────────────────────────────────────────────────
    cost_per_call: float | None = None,    # explicit USD cost or price/call calculation

    # ── Policy & Canary ───────────────────────────────────────────────────────
    policy: Callable | None = None,        # policy(prompt) -> "allow"|"block"|"flag"
    canary: dict | None = None,            # {"fn": callable, "weight": 0.0-1.0}
) -> Callable:
    """
    Decorator factory that wraps an LLM call function with lensllm features.

    Example:
        @track(
            cache=True,
            retries=3,
            cost_per_call=0.0005,  # price per call / 1M tokens calculation
            canary={"fn": call_gpt4, "weight": 0.05},
        )
        def call_gemini(prompt: str) -> str:
            return gemini_client.generate(prompt)
    """
    cfg = get_config()

    # ── Cache config ──────────────────────────────────────────────────────────
    _cache_enabled = cache if cache is not None else cfg.cache_enabled
    _threshold     = similarity_threshold if similarity_threshold is not None else cfg.similarity_threshold
    _ttl           = cache_ttl if cache_ttl is not None else cfg.cache_ttl

    # ── Retry config ──────────────────────────────────────────────────────────
    _retries    = retries    if retries    is not None else cfg.retries
    _base_delay = base_delay if base_delay is not None else cfg.base_delay
    _max_delay  = max_delay  if max_delay  is not None else cfg.max_delay
    _jitter     = jitter     if jitter     is not None else cfg.jitter
    _retry_on   = retry_on   if retry_on   is not None else DEFAULT_RETRY_ON
    _fallbacks  = fallback or []

    # ── Cost config ───────────────────────────────────────────────────────────
    _cost_per_call = cost_per_call

    # ── Policy config ─────────────────────────────────────────────────────────
    _policy     = policy

    # ── Canary config ─────────────────────────────────────────────────────────
    _canary_config = parse_canary(canary)

    def decorator(fn: Callable) -> Callable:

        @functools.wraps(fn)
        def wrapper(prompt: str, *args: Any, **kwargs: Any) -> Any:
            # ── 0. Ensure DB schema ───────────────────────────────────────────
            init_db()

            cache_store = get_cache_backend(cfg.db_path)
            log_store   = get_metrics_store(cfg.db_path)

            # ── 1. Trace / span IDs (distributed tracing) ────────────────────
            parent_trace = get_current_trace()    # (trace_id, span_id) or None
            if parent_trace:
                trace_id       = parent_trace[0]
                parent_span_id = parent_trace[1]
            else:
                trace_id       = str(uuid.uuid4())
                parent_span_id = None

            span_id   = str(uuid.uuid4())
            timestamp = time.time()

            # Set ContextVar so nested @track calls inherit this trace
            trace_token = set_current_trace(trace_id, span_id)

            try:
                return _wrapper_body(
                    fn=fn,
                    prompt=prompt,
                    args=args,
                    kwargs=kwargs,
                    cfg=cfg,
                    cache_store=cache_store,
                    log_store=log_store,
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    timestamp=timestamp,
                    _cache_enabled=_cache_enabled,
                    _threshold=_threshold,
                    _ttl=_ttl,
                    _retries=_retries,
                    _base_delay=_base_delay,
                    _max_delay=_max_delay,
                    _jitter=_jitter,
                    _retry_on=_retry_on,
                    _fallbacks=_fallbacks,
                    _cost_per_call=_cost_per_call,
                    _policy=_policy,
                    _canary_config=_canary_config,
                    rate_limit_rate=rate_limit_rate,
                    rate_limit_tokens=rate_limit_tokens,
                    max_concurrency=max_concurrency,
                )
            finally:
                # Restore previous trace context
                reset_current_trace(trace_token)

        wrapper._gateway_tracked = True
        return wrapper

    return decorator


def _wrapper_body(
    fn, prompt, args, kwargs, cfg,
    cache_store, log_store,
    trace_id, span_id, parent_span_id, timestamp,
    _cache_enabled, _threshold, _ttl,
    _retries, _base_delay, _max_delay, _jitter, _retry_on, _fallbacks,
    _cost_per_call, _policy, _canary_config,
    rate_limit_rate, rate_limit_tokens, max_concurrency,
) -> Any:
    """Inner logic extracted so the try/finally trace token reset is clean."""

    # ── 1. Policy hook ────────────────────────────────────────────────────────
    if _policy is not None:
        decision = _policy(prompt)
        if decision == "block":
            raise PolicyViolationError(
                f"Policy hook blocked the request for function '{fn.__name__}'."
            )

    # ── 2. Embed the prompt ───────────────────────────────────────────────────
    prompt_embedding = embed(prompt)

    # ── 3. Cache lookup ───────────────────────────────────────────────────────
    if _cache_enabled:
        cached = cache_store.lookup(prompt_embedding, _threshold)
        if cached is not None:
            hit_latency_ms = (time.time() - timestamp) * 1000
            log_store.log_request(
                LogEntry(
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    function_name=fn.__name__,
                    prompt=prompt,
                    response=cached.response,
                    cache_hit=True,
                    latency_ms=0.0,
                    cost_estimate=0.0,
                    timestamp=timestamp,
                )
            )
            _last_call_info.set(CallInfo(
                cache_hit=True,
                latency_ms=hit_latency_ms,
                fn_name=fn.__name__,
            ))
            return cached.response

    # ── 4. Rate limiter (Phase 4) ─────────────────────────────────────────────
    if rate_limit_rate is not None or rate_limit_tokens is not None:
        _rate   = rate_limit_rate   if rate_limit_rate   is not None else cfg.rate_limit_rate
        _tokens = rate_limit_tokens if rate_limit_tokens is not None else cfg.rate_limit_max_tokens
        limiter = get_limiter(rate=_rate, max_tokens=_tokens)
        limiter.acquire(block=True)

    # ── 5. Select primary vs canary function ─────────────────────────────────
    call_fn = fn
    variant: Optional[str] = None
    if _canary_config and should_use_canary(_canary_config.weight):
        call_fn  = _canary_config.fn
        variant  = "canary"
    elif _canary_config:
        variant  = "primary"

    # ── 6. Concurrency queue ──────────────────────────────────────────────────
    _use_queue = max_concurrency is not None
    _queue = get_queue(max_concurrency) if _use_queue else None

    def _call_with_retry():
        return retry_with_backoff(
            call_fn,
            args=(prompt,) + args,
            kwargs=kwargs,
            max_retries=_retries,
            base_delay=_base_delay,
            max_delay=_max_delay,
            jitter=_jitter,
            retry_on=_retry_on,
        )

    # ── 7. Call the wrapped function ─────────────────────────────────────────
    start = time.perf_counter()
    response: Any = None
    error_msg: Optional[str] = None
    succeeded_via: str = call_fn.__name__

    try:
        if _use_queue and _queue:
            with _queue.slot():
                response = _call_with_retry()
        else:
            response = _call_with_retry()

    except Exception as primary_exc:
        if _fallbacks:
            chain = FallbackChain(_fallbacks)
            result = chain.run(prompt, *args, **kwargs)
            if result.succeeded:
                response = result.response
                succeeded_via = result.succeeded_with
                error_msg = (
                    f"primary failed; fallback succeeded via {succeeded_via}. "
                    f"Chain: {result.attempts}"
                )
            else:
                error_msg = (
                    f"primary failed; all fallbacks failed. "
                    f"Chain: {result.attempts}"
                )
                latency_ms = (time.perf_counter() - start) * 1000
                log_store.log_request(LogEntry(
                    trace_id=trace_id, span_id=span_id,
                    parent_span_id=parent_span_id,
                    function_name=fn.__name__,
                    prompt=prompt, response=None,
                    cache_hit=False,
                    latency_ms=round(latency_ms, 2),
                    cost_estimate=None, timestamp=timestamp,
                    error=error_msg, variant=variant,
                ))
                raise primary_exc
        else:
            latency_ms = (time.perf_counter() - start) * 1000
            log_store.log_request(LogEntry(
                trace_id=trace_id, span_id=span_id,
                parent_span_id=parent_span_id,
                function_name=fn.__name__,
                prompt=prompt, response=None,
                cache_hit=False,
                latency_ms=round(latency_ms, 2),
                cost_estimate=None, timestamp=timestamp,
                error=str(primary_exc), variant=variant,
            ))
            raise

    # ── 8. Streaming passthrough ──────────────────────────────────────────────
    if is_generator(response):
        import types as _types
        # Build the on_complete callback (closure over all log/cache context)
        def _on_stream_complete(full_text: str, latency_ms: float) -> None:
            _finish_log_and_cache(
                log_store=log_store,
                cache_store=cache_store,
                trace_id=trace_id, span_id=span_id,
                parent_span_id=parent_span_id,
                fn_name=succeeded_via,
                prompt=prompt,
                prompt_embedding=None,   # re-embed if needed; skip for streams
                response_text=full_text,
                latency_ms=latency_ms,
                timestamp=timestamp,
                variant=variant,
                error_msg=error_msg,
                cost_per_call=_cost_per_call,
                cache_enabled=_cache_enabled,
                ttl=_ttl,
            )

        if isinstance(response, _types.AsyncGeneratorType):
            from lensllm.streaming import wrap_async_stream
            return wrap_async_stream(response, _on_stream_complete, start)
        else:
            return wrap_stream(response, _on_stream_complete, start)

    # ── 9. Log + cache (non-streaming) ───────────────────────────────────────
    latency_ms = (time.perf_counter() - start) * 1000
    response_text = str(response) if response is not None else ""

    _finish_log_and_cache(
        log_store=log_store,
        cache_store=cache_store,
        trace_id=trace_id, span_id=span_id,
        parent_span_id=parent_span_id,
        fn_name=succeeded_via,
        prompt=prompt,
        prompt_embedding=embed(prompt),
        response_text=response_text,
        latency_ms=round(latency_ms, 2),
        timestamp=timestamp,
        variant=variant,
        error_msg=error_msg if "fallback succeeded" in (error_msg or "") else None,
        cost_per_call=_cost_per_call,
        cache_enabled=_cache_enabled,
        ttl=_ttl,
    )

    # ── 10. Record call metadata for get_last_call_info() ─────────────────────
    _last_call_info.set(CallInfo(
        cache_hit=False,
        latency_ms=round(latency_ms, 2),
        fn_name=fn.__name__,
    ))

    return response


# ── Shared helpers ────────────────────────────────────────────────────────────

def _finish_log_and_cache(
    log_store, cache_store,
    trace_id, span_id, parent_span_id, fn_name,
    prompt, prompt_embedding,
    response_text, latency_ms, timestamp,
    variant, error_msg, cost_per_call,
    cache_enabled, ttl,
) -> None:
    """Write log entry and cache entry for a completed (non-streaming) call."""
    if cost_per_call is not None:
        cost_estimate = round(cost_per_call, 8)
    else:
        estimated_tokens = (len(prompt) + len(response_text)) / 4
        cost_estimate = round(estimated_tokens * 0.000002, 8)

    log_store.log_request(LogEntry(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        function_name=fn_name,
        prompt=prompt,
        response=response_text,
        cache_hit=False,
        latency_ms=latency_ms,
        cost_estimate=cost_estimate,
        timestamp=timestamp,
        error=error_msg,
        variant=variant,
    ))

    if cache_enabled and response_text and prompt_embedding is not None:
        cache_store.write(
            prompt=prompt,
            prompt_embedding=prompt_embedding,
            response=response_text,
            ttl=ttl,
        )
