"""
tests.test_storage_backends
============================
Unit & Integration tests for Storage Abstraction Layer (SQLite and Postgres).
"""

import os
import time
import pytest
import numpy as np

from gateway_sdk.backends.base import CacheEntry
from gateway_sdk.backends.sqlite_backend import SQLiteCacheBackend, SQLiteMetricsStore
from gateway_sdk.backends.factory import get_cache_backend, get_metrics_store, reset_backends
from gateway_sdk.logging.store import LogEntry


@pytest.fixture
def tmp_sqlite_db(tmp_path):
    db_file = str(tmp_path / "test_gateway.db")
    from gateway_sdk.db import init_db
    init_db(db_file)
    return db_file


def test_sqlite_cache_backend(tmp_sqlite_db):
    cache = SQLiteCacheBackend(db_path=tmp_sqlite_db)
    assert cache.count() == 0

    emb = np.random.rand(384).astype(np.float32)
    emb /= np.linalg.norm(emb)

    # Write entry
    cache.write("What is Python?", emb, "Python is a programming language.", ttl=60)
    assert cache.count() == 1

    # Exact match lookup
    hit = cache.lookup(emb, threshold=0.90)
    assert hit is not None
    assert hit.prompt == "What is Python?"
    assert hit.response == "Python is a programming language."
    assert hit.similarity >= 0.99

    # No match for orthogonal vector
    diff_emb = -emb
    miss = cache.lookup(diff_emb, threshold=0.90)
    assert miss is None


def test_sqlite_metrics_store(tmp_sqlite_db):
    store = SQLiteMetricsStore(db_path=tmp_sqlite_db)
    entry = LogEntry(
        trace_id="t-123",
        span_id="s-456",
        parent_span_id=None,
        function_name="ask_gemini",
        prompt="Hello world",
        response="Hi there!",
        cache_hit=False,
        latency_ms=120.0,
        cost_estimate=0.00005,
        timestamp=time.time(),
        variant="primary",
    )
    store.log_request(entry)

    stats = store.get_stats()
    assert stats["total_calls"] == 1
    assert stats["cache_misses"] == 1
    assert stats["total_cost_usd"] == 0.00005

    recent = store.get_recent_logs(limit=10)
    assert len(recent) == 1
    assert recent[0]["trace_id"] == "t-123"

    traces = store.get_traces(limit=10)
    assert len(traces) == 1
    assert traces[0]["trace_id"] == "t-123"

    cost = store.get_cost_since(time.time() - 3600)
    assert cost == 0.00005


def test_factory_backend_switching(monkeypatch):
    reset_backends()
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.delenv("GATEWAY_SDK_POSTGRES_URL", raising=False)

    cache = get_cache_backend()
    metrics = get_metrics_store()

    assert isinstance(cache, SQLiteCacheBackend)
    assert isinstance(metrics, SQLiteMetricsStore)

    reset_backends()
