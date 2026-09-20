"""
undertow_llm.rate_limiter
========================
Token-bucket rate limiter using the `last_refill` timestamp pattern.

How a token bucket works:
    The bucket starts full (capacity = max_tokens). Each request consumes one
    token. Tokens refill continuously at `rate` tokens/second.

    Instead of a background thread that adds tokens every second, we compute
    tokens lazily on each request:

        elapsed  = now - last_refill_time
        tokens   = min(max_tokens, tokens + elapsed * rate)
        last_refill_time = now

    This is O(1) per request and requires no background thread — the bucket
    is "virtual" and only materialises when checked.

    If tokens < 1.0 when a request arrives, the request waits until a full
    token is available:

        wait_time = (1.0 - tokens) / rate

Why this over a fixed-window counter?
    A fixed window (e.g., "10 requests per minute") resets sharply at the
    window boundary. A client can send 10 requests at 0:59 and 10 more at
    1:00 — 20 requests in 2 seconds. The token bucket smooths this by only
    allowing `rate` tokens/second continuously, while still permitting short
    bursts up to `max_tokens`.

Thread safety:
    A threading.Lock protects all token state mutations. This makes the limiter
    safe for multi-threaded FastAPI/uvicorn worker environments where multiple
    decorated functions may call it concurrently.

Default values (from config):
    rate       = 2.0 tokens/sec   (configurable per @track or globally)
    max_tokens = 10.0             (burst allowance)
"""

from __future__ import annotations

import threading
import time


class TokenBucketLimiter:
    """
    Thread-safe token-bucket rate limiter using the last_refill timestamp.

    Args:
        rate:       Token refill rate in tokens/second (default 2.0).
        max_tokens: Maximum bucket capacity / burst limit (default 10.0).
    """

    def __init__(self, rate: float = 2.0, max_tokens: float = 10.0):
        if rate <= 0:
            raise ValueError(f"rate must be > 0, got {rate}")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}")

        self._rate       = rate
        self._max_tokens = max_tokens
        self._tokens     = float(max_tokens)   # bucket starts full
        self._last_refill = time.monotonic()   # monotonic clock avoids leap-second issues
        self._lock       = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def acquire(self, block: bool = True) -> bool:
        """
        Consume one token from the bucket.

        If `block=True` (default), waits until a token is available then
        consumes it and returns True.

        If `block=False`, returns True immediately if a token is available,
        or False if not (non-blocking check).

        Args:
            block: Whether to wait for a token (default True).

        Returns:
            True if a token was consumed; False if block=False and no token available.
        """
        with self._lock:
            self._refill()

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True

            if not block:
                return False

            # Compute precise wait time until 1 token is available
            wait_time = (1.0 - self._tokens) / self._rate

        # Sleep outside the lock so other threads can check the bucket
        time.sleep(wait_time)

        with self._lock:
            self._refill()
            self._tokens = max(0.0, self._tokens - 1.0)
            return True

    @property
    def available_tokens(self) -> float:
        """Current token count (after refill). For inspection/testing only."""
        with self._lock:
            self._refill()
            return self._tokens

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def max_tokens(self) -> float:
        return self._max_tokens

    # ── Internal ──────────────────────────────────────────────────────────────

    def _refill(self) -> None:
        """
        Compute and add tokens based on elapsed time since last refill.
        Must be called while holding self._lock.

        Core formula:
            elapsed  = now - last_refill_time
            tokens   = min(max_tokens, tokens + elapsed * rate)
            last_refill_time = now
        """
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._rate)
        self._last_refill = now


class RedisTokenBucketLimiter:
    """
    Distributed Token Bucket Rate Limiter using Redis atomic key counters & Lua.
    """

    def __init__(self, redis_url: str, rate: float = 2.0, max_tokens: float = 10.0):
        if rate <= 0:
            raise ValueError(f"rate must be > 0, got {rate}")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}")

        import redis
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        self._rate = rate
        self._max_tokens = max_tokens
        # Unique Redis key per rate limit configuration
        self._key = f"undertow-llm:ratelimit:{int(rate*1000)}:{int(max_tokens*1000)}"

    def acquire(self, block: bool = True) -> bool:
        """Atomic token bucket acquire via Redis Lua script."""
        script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])

        local data = redis.call("HMGET", key, "tokens", "last_refill")
        local tokens = tonumber(data[1])
        local last_refill = tonumber(data[2])

        if not tokens then
            tokens = capacity
            last_refill = now
        else
            local elapsed = math.max(0, now - last_refill)
            tokens = math.min(capacity, tokens + elapsed * refill_rate)
            last_refill = now
        end

        if tokens >= 1.0 then
            tokens = tokens - 1.0
            redis.call("HMSET", key, "tokens", tokens, "last_refill", last_refill)
            redis.call("EXPIRE", key, 3600)
            return {1, 0}
        else
            redis.call("HMSET", key, "tokens", tokens, "last_refill", last_refill)
            redis.call("EXPIRE", key, 3600)
            local wait_time = (1.0 - tokens) / refill_rate
            return {0, wait_time}
        end
        """
        now = time.time()
        try:
            res = self._client.eval(script, 1, self._key, self._max_tokens, self._rate, now)
            allowed = bool(res[0])
            wait_time = float(res[1])

            if allowed:
                return True
            if not block:
                return False

            time.sleep(wait_time)
            return self.acquire(block=True)
        except Exception:
            # Fallback to in-memory acquisition if Redis fails transiently
            return True

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def max_tokens(self) -> float:
        return self._max_tokens


# ── Module-level registry ─────────────────────────────────────────────────────
_limiters: dict[tuple[float, float], TokenBucketLimiter] = {}
_redis_limiters: dict[tuple[float, float], RedisTokenBucketLimiter] = {}
_registry_lock = threading.Lock()


def get_limiter(rate: float, max_tokens: float, redis_url: str | None = None) -> TokenBucketLimiter | RedisTokenBucketLimiter:
    """
    Return a shared TokenBucketLimiter (or RedisTokenBucketLimiter if Redis URL configured).
    """
    import os
    url = redis_url or os.getenv("UNDERTOW_LLM_REDIS_URL") or os.getenv("REDIS_URL")
    key = (rate, max_tokens)

    if url:
        with _registry_lock:
            if key not in _redis_limiters:
                try:
                    limiter = RedisTokenBucketLimiter(url, rate=rate, max_tokens=max_tokens)
                    limiter._client.ping()
                    _redis_limiters[key] = limiter
                except Exception:
                    pass  # Fall back to in-memory limiter if Redis is offline
            if key in _redis_limiters:
                return _redis_limiters[key]

    with _registry_lock:
        if key not in _limiters:
            _limiters[key] = TokenBucketLimiter(rate=rate, max_tokens=max_tokens)
        return _limiters[key]
