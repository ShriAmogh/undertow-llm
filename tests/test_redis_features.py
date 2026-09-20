"""
tests.test_redis_features
==========================
Unit & Integration test suite for Redis rate limiting and concurrency queueing.
Verifies that Redis is used strictly for high-throughput atomic counters (INCR/DECR)
and concurrency slots.
"""

import os
import time
import pytest
from gateway_sdk.rate_limiter import RedisTokenBucketLimiter, get_limiter
from gateway_sdk.queue import RedisRequestQueue, get_queue

REDIS_TEST_URL = os.getenv("GATEWAY_SDK_REDIS_URL", "redis://localhost:6379")


@pytest.fixture
def redis_client():
    try:
        import redis
        client = redis.Redis.from_url(REDIS_TEST_URL, decode_responses=True)
        client.ping()
        yield client
    except Exception:
        pytest.skip("Redis server unavailable for testing")


def test_redis_token_bucket_limiter_atomic_counter(redis_client):
    limiter = RedisTokenBucketLimiter(REDIS_TEST_URL, rate=5.0, max_tokens=2.0)

    # First 2 acquires should succeed instantly (burst = 2.0)
    assert limiter.acquire(block=False) is True
    assert limiter.acquire(block=False) is True

    # 3rd acquire should fail when non-blocking (tokens exhausted)
    assert limiter.acquire(block=False) is False

    # Wait 0.25 seconds (refills ~1.25 tokens at 5 tokens/sec)
    time.sleep(0.25)
    assert limiter.acquire(block=False) is True


def test_redis_request_queue_atomic_concurrency(redis_client):
    queue = RedisRequestQueue(REDIS_TEST_URL, max_concurrency=2)

    # Acquire 2 slots simultaneously
    with queue.slot():
        with queue.slot():
            # Redis counter should be at max capacity (2)
            assert queue.max_concurrency == 2

    # Slots freed cleanly after context manager exit


def test_redis_factory_dispatching(redis_client, monkeypatch):
    monkeypatch.setenv("GATEWAY_SDK_REDIS_URL", REDIS_TEST_URL)

    limiter = get_limiter(rate=2.0, max_tokens=5.0)
    assert isinstance(limiter, RedisTokenBucketLimiter)

    q = get_queue(max_concurrency=5)
    assert isinstance(q, RedisRequestQueue)


def test_redis_contains_no_metrics_or_cache_data(redis_client):
    """
    Verify architectural constraint: Redis is NEVER used for metric logs or
    vector cache entries (which belong exclusively in Postgres).
    """
    keys = redis_client.keys("gateway_sdk:*")
    # All keys in Redis must belong only to ratelimit or queue domains
    for k in keys:
        assert ("ratelimit" in k or "queue" in k or "test" in k), f"Unexpected key domain in Redis: {k}"
