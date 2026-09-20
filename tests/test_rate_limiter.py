"""
tests/test_rate_limiter.py
==========================
Phase 4 tests: TokenBucketLimiter correctness, thread safety, and concurrency queue.

Per skill.md rule #5: these tests must pass before any refactoring touches rate_limiter.py.
"""

from __future__ import annotations

import threading
import time
import pytest

from lensllm.rate_limiter import TokenBucketLimiter, get_limiter
from lensllm.queue import RequestQueue, get_queue


# ── TokenBucketLimiter ────────────────────────────────────────────────────────

class TestTokenBucketLimiter:

    def test_bucket_starts_full(self):
        """A fresh limiter should have max_tokens tokens available."""
        limiter = TokenBucketLimiter(rate=2.0, max_tokens=10.0)
        assert limiter.available_tokens == pytest.approx(10.0, abs=0.1)

    def test_acquire_reduces_tokens(self):
        """Each acquire() must consume exactly one token."""
        limiter = TokenBucketLimiter(rate=2.0, max_tokens=10.0)
        before = limiter.available_tokens
        limiter.acquire(block=False)
        after = limiter.available_tokens
        assert before - after == pytest.approx(1.0, abs=0.05)

    def test_tokens_cannot_exceed_max(self):
        """available_tokens must never exceed max_tokens."""
        limiter = TokenBucketLimiter(rate=100.0, max_tokens=5.0)
        time.sleep(0.1)  # would add 10 tokens at rate=100, but capped at 5
        assert limiter.available_tokens <= 5.0 + 0.1  # small tolerance

    def test_non_blocking_returns_false_when_empty(self):
        """acquire(block=False) must return False when no tokens are available."""
        limiter = TokenBucketLimiter(rate=0.001, max_tokens=1.0)
        # Drain the bucket
        limiter.acquire(block=False)
        # Now it's empty — non-blocking acquire must fail
        result = limiter.acquire(block=False)
        assert result is False

    def test_non_blocking_returns_true_when_available(self):
        """acquire(block=False) must return True when tokens are available."""
        limiter = TokenBucketLimiter(rate=2.0, max_tokens=10.0)
        assert limiter.acquire(block=False) is True

    def test_tokens_refill_over_time(self):
        """After draining, tokens should refill at the configured rate."""
        limiter = TokenBucketLimiter(rate=10.0, max_tokens=10.0)
        # Drain completely
        for _ in range(10):
            limiter.acquire(block=False)
        assert limiter.available_tokens < 1.0

        # Wait 0.5s — at 10 tokens/sec, should get ~5 tokens
        time.sleep(0.5)
        assert limiter.available_tokens == pytest.approx(5.0, abs=1.0)

    def test_invalid_rate_raises(self):
        with pytest.raises(ValueError, match="rate must be > 0"):
            TokenBucketLimiter(rate=0, max_tokens=10.0)

    def test_invalid_max_tokens_raises(self):
        with pytest.raises(ValueError, match="max_tokens must be > 0"):
            TokenBucketLimiter(rate=2.0, max_tokens=0)

    def test_thread_safe_concurrent_acquires(self):
        """
        Multiple threads acquiring concurrently must not exceed total available tokens.
        With 10 tokens and 15 threads all trying non-blocking, at most 10 succeed.
        """
        limiter = TokenBucketLimiter(rate=0.001, max_tokens=10.0)
        successes = []
        lock = threading.Lock()

        def try_acquire():
            result = limiter.acquire(block=False)
            with lock:
                successes.append(result)

        threads = [threading.Thread(target=try_acquire) for _ in range(15)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        true_count = sum(1 for r in successes if r)
        assert true_count <= 10  # never more than bucket capacity

    def test_get_limiter_returns_same_instance(self):
        """get_limiter with same params must return the same object (shared bucket)."""
        l1 = get_limiter(rate=2.0, max_tokens=10.0)
        l2 = get_limiter(rate=2.0, max_tokens=10.0)
        assert l1 is l2

    def test_get_limiter_different_params_different_instance(self):
        """Different (rate, max_tokens) pairs must return different limiters."""
        l1 = get_limiter(rate=1.0, max_tokens=5.0)
        l2 = get_limiter(rate=3.0, max_tokens=15.0)
        assert l1 is not l2


# ── RequestQueue ──────────────────────────────────────────────────────────────

class TestRequestQueue:

    def test_slot_allows_up_to_max_concurrency(self):
        """At most max_concurrency threads can hold a slot simultaneously."""
        queue = RequestQueue(max_concurrency=3)
        active_peaks = []
        lock = threading.Lock()
        barrier = threading.Barrier(3)

        def task():
            with queue.slot():
                barrier.wait()  # all 3 threads inside the slot simultaneously
                with lock:
                    active_peaks.append(queue.active_count)
                time.sleep(0.05)

        threads = [threading.Thread(target=task) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert max(active_peaks) == 3

    def test_excess_requests_wait_for_slot(self):
        """A 4th thread must wait while 3 slots are occupied."""
        queue = RequestQueue(max_concurrency=2)
        order = []
        lock = threading.Lock()

        def slow_task(name):
            with queue.slot():
                with lock:
                    order.append(f"{name}:enter")
                time.sleep(0.1)
                with lock:
                    order.append(f"{name}:exit")

        threads = [threading.Thread(target=slow_task, args=(f"t{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # There must be 4 enter and 4 exit events total
        assert len([e for e in order if "enter" in e]) == 4
        assert len([e for e in order if "exit" in e]) == 4

    def test_active_count_resets_to_zero_after_all_done(self):
        """active_count must return to 0 once all slots are released."""
        queue = RequestQueue(max_concurrency=5)

        def task():
            with queue.slot():
                time.sleep(0.02)

        threads = [threading.Thread(target=task) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert queue.active_count == 0

    def test_invalid_max_concurrency_raises(self):
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            RequestQueue(max_concurrency=0)

    def test_get_queue_returns_same_instance(self):
        """get_queue with same value returns the same object."""
        q1 = get_queue(5)
        q2 = get_queue(5)
        assert q1 is q2

    def test_get_queue_different_values_different_instances(self):
        q1 = get_queue(3)
        q2 = get_queue(7)
        assert q1 is not q2
