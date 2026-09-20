"""
tests/test_logging.py
=====================
Phase 2 tests: request logging — LogStore persistence and stats aggregation.
"""

from __future__ import annotations

import time
import uuid
import pytest

from gateway_sdk.logging.store import LogStore, LogEntry


@pytest.fixture
def log_store(temp_db):
    return LogStore(db_path=temp_db)


def make_entry(
    *,
    cache_hit: bool = False,
    latency_ms: float = 500.0,
    cost: float = 0.0001,
    error: str | None = None,
    function_name: str = "call_llm",
    prompt: str = "test prompt",
    response: str = "test response",
) -> LogEntry:
    return LogEntry(
        trace_id=str(uuid.uuid4()),
        span_id=str(uuid.uuid4()),
        function_name=function_name,
        prompt=prompt,
        response=response,
        cache_hit=cache_hit,
        latency_ms=latency_ms,
        cost_estimate=cost,
        timestamp=time.time(),
        error=error,
    )


class TestLogStore:

    def test_log_request_persists(self, log_store):
        """A logged entry must appear in get_recent_logs()."""
        entry = make_entry(cache_hit=False, latency_ms=312.5)
        log_store.log_request(entry)

        logs = log_store.get_recent_logs(limit=10)
        assert len(logs) == 1
        assert logs[0]["function_name"] == "call_llm"
        assert logs[0]["cache_hit"] == 0
        assert logs[0]["latency_ms"] == pytest.approx(312.5)

    def test_stats_empty_db(self, log_store):
        """Stats on an empty DB should return zeros, not errors."""
        stats = log_store.get_stats()
        assert stats["total_calls"] == 0
        assert stats["hit_rate_pct"] == 0.0
        assert stats["avg_latency_ms"] == 0.0

    def test_stats_hit_rate(self, log_store):
        """hit_rate_pct = cache_hits / total_calls * 100."""
        # 3 misses, 2 hits → 40% hit rate
        for _ in range(3):
            log_store.log_request(make_entry(cache_hit=False, latency_ms=200.0))
        for _ in range(2):
            log_store.log_request(make_entry(cache_hit=True, latency_ms=0.0))

        stats = log_store.get_stats()
        assert stats["total_calls"] == 5
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 3
        assert stats["hit_rate_pct"] == pytest.approx(40.0)

    def test_stats_avg_latency_excludes_hits(self, log_store):
        """avg_latency_ms must only average miss calls (hits have 0ms, not meaningful)."""
        log_store.log_request(make_entry(cache_hit=False, latency_ms=100.0))
        log_store.log_request(make_entry(cache_hit=False, latency_ms=200.0))
        log_store.log_request(make_entry(cache_hit=True,  latency_ms=0.0))   # should not affect avg

        stats = log_store.get_stats()
        assert stats["avg_latency_ms"] == pytest.approx(150.0)

    def test_stats_total_cost(self, log_store):
        """total_cost_usd must sum all cost_estimate values."""
        log_store.log_request(make_entry(cost=0.001))
        log_store.log_request(make_entry(cost=0.002))
        stats = log_store.get_stats()
        assert stats["total_cost_usd"] == pytest.approx(0.003, rel=1e-5)

    def test_stats_error_count(self, log_store):
        """Errors are counted and reported as a rate."""
        log_store.log_request(make_entry(error="Timeout"))
        log_store.log_request(make_entry(error=None))
        stats = log_store.get_stats()
        assert stats["error_count"] == 1
        assert stats["error_rate_pct"] == pytest.approx(50.0)

    def test_recent_logs_ordered_newest_first(self, log_store):
        """get_recent_logs must return entries newest-first."""
        for i in range(5):
            e = make_entry(prompt=f"prompt {i}")
            e.timestamp = time.time() + i   # force ordering
            log_store.log_request(e)

        logs = log_store.get_recent_logs(limit=5)
        timestamps = [l["timestamp"] for l in logs]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_recent_logs_limit(self, log_store):
        """get_recent_logs must respect the limit parameter."""
        for _ in range(20):
            log_store.log_request(make_entry())
        assert len(log_store.get_recent_logs(limit=5)) == 5

    def test_get_trends_returns_list(self, log_store):
        """get_trends() must return a list (empty if no data)."""
        trends = log_store.get_trends()
        assert isinstance(trends, list)

    def test_get_trends_buckets(self, log_store):
        """Trends must bucket requests into correct time slots."""
        log_store.log_request(make_entry(cache_hit=False))
        log_store.log_request(make_entry(cache_hit=True))

        trends = log_store.get_trends(bucket_minutes=5, limit_buckets=24)
        # Should have at least one bucket with our two requests
        total = sum(t["total_calls"] for t in trends)
        assert total == 2
