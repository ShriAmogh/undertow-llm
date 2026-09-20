"""
lensllm.queue
=================
Async concurrency limiter — bounds simultaneous in-flight LLM calls.

Problem it solves:
    Without a concurrency limit, a burst of 50 concurrent requests all hit
    the LLM provider simultaneously. Even if the rate limiter spaces them out
    at the entry point, async tasks can still pile up if each call is slow
    (e.g., a 30s streaming response).

    `RequestQueue` bounds how many calls can be executing at the same time.
    Excess requests wait in an asyncio queue rather than hammering the provider.

Design:
    - Uses asyncio.Semaphore — the idiomatic Python tool for bounding
      concurrency. A semaphore with value N allows N coroutines to hold it
      simultaneously; the (N+1)th must wait.
    - The semaphore is stored per (max_concurrency) value — shared across
      all @track calls with the same concurrency setting.
    - For sync functions wrapped by @track, concurrency limiting applies
      conceptually at the decorator level (blocking acquire before call).

Technical learning — asyncio.Semaphore vs threading.Semaphore:
    asyncio.Semaphore uses cooperative multitasking — it yields control to
    the event loop while waiting, so other coroutines can run. This is far
    more efficient than a threading.Semaphore which blocks the OS thread.
    Use asyncio.Semaphore in async contexts, threading.Semaphore in sync.

    For the MVP (sync decorator), we use threading.Semaphore so it works
    in any calling context without requiring async/await.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager


class RequestQueue:
    """
    Thread-safe concurrency limiter using threading.Semaphore.

    Limits the number of simultaneous LLM calls. When the limit is reached,
    excess calls block until a slot frees up.

    Args:
        max_concurrency: Maximum number of simultaneous in-flight calls.
    """

    def __init__(self, max_concurrency: int = 10):
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        self._semaphore = threading.Semaphore(max_concurrency)
        self._max = max_concurrency
        self._active = 0
        self._lock = threading.Lock()

    @contextmanager
    def slot(self):
        """
        Context manager that acquires a concurrency slot before entering
        the block and releases it on exit.

        Usage:
            queue = RequestQueue(max_concurrency=5)
            with queue.slot():
                response = call_llm(prompt)  # at most 5 of these run at once
        """
        acquired = self._semaphore.acquire(timeout=300)  # 5-min timeout
        if not acquired:
            raise TimeoutError("Request queue timeout: no slot available within 5 minutes")
        with self._lock:
            self._active += 1
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
            self._semaphore.release()

    @property
    def active_count(self) -> int:
        """Number of requests currently in-flight."""
        with self._lock:
            return self._active

    @property
    def max_concurrency(self) -> int:
        return self._max


class RedisRequestQueue:
    """
    Distributed concurrency queue using Redis atomic INCR / DECR counters.
    """

    def __init__(self, redis_url: str, max_concurrency: int = 10):
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        import redis
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        self._max = max_concurrency
        self._key = f"lensllm:queue:active:{max_concurrency}"

    @contextmanager
    def slot(self):
        """Acquire distributed slot counter in Redis."""
        import time
        start = time.time()
        timeout = 300
        acquired = False
        while time.time() - start < timeout:
            try:
                active = self._client.incr(self._key)
                self._client.expire(self._key, 300)
                if active <= self._max:
                    acquired = True
                    break
                else:
                    self._client.decr(self._key)
                    time.sleep(0.05)
            except Exception:
                acquired = True
                break

        if not acquired:
            raise TimeoutError("Request queue timeout: no slot available within 5 minutes")

        try:
            yield
        finally:
            if acquired:
                try:
                    self._client.decr(self._key)
                except Exception:
                    pass

    @property
    def max_concurrency(self) -> int:
        return self._max


# ── Module-level registry ─────────────────────────────────────────────────────
_queues: dict[int, RequestQueue] = {}
_redis_queues: dict[int, RedisRequestQueue] = {}
_queue_lock = threading.Lock()


def get_queue(max_concurrency: int, redis_url: str | None = None) -> RequestQueue | RedisRequestQueue:
    """Return a shared RequestQueue (or RedisRequestQueue if Redis URL configured)."""
    import os
    url = redis_url or os.getenv("LENSLLM_REDIS_URL") or os.getenv("REDIS_URL")

    if url:
        with _queue_lock:
            if max_concurrency not in _redis_queues:
                try:
                    q = RedisRequestQueue(url, max_concurrency=max_concurrency)
                    q._client.ping()
                    _redis_queues[max_concurrency] = q
                except Exception:
                    pass  # Fall back to in-memory queue if Redis is offline
            if max_concurrency in _redis_queues:
                return _redis_queues[max_concurrency]

    with _queue_lock:
        if max_concurrency not in _queues:
            _queues[max_concurrency] = RequestQueue(max_concurrency)
        return _queues[max_concurrency]
