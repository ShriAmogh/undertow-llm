"""
tests.test_postgres_integration
================================
Integration test suite for PostgresCacheBackend (pgvector) and PostgresMetricsStore.
Requires a running PostgreSQL instance with pgvector (e.g. via docker compose up).
"""

import os
import time
import uuid
import pytest
import numpy as np

from gateway_sdk.backends.postgres_backend import PostgresCacheBackend, PostgresMetricsStore
from gateway_sdk.backends.factory import get_cache_backend, get_metrics_store, reset_backends
from gateway_sdk.logging.store import LogEntry
from gateway_sdk.decorator import track, get_last_call_info

POSTGRES_TEST_URL = os.getenv(
    "GATEWAY_SDK_POSTGRES_URL",
    "postgresql://gateway:gateway@localhost:5432/gateway"
)


@pytest.fixture
def setup_postgres_env(monkeypatch):
    monkeypatch.setenv("GATEWAY_SDK_POSTGRES_URL", POSTGRES_TEST_URL)
    from gateway_sdk.config import configure
    configure(postgres_url=POSTGRES_TEST_URL)
    reset_backends()
    yield
    monkeypatch.delenv("GATEWAY_SDK_POSTGRES_URL", raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    configure(postgres_url=None)
    reset_backends()


def test_postgres_cache_backend_vector_similarity():
    cache = PostgresCacheBackend(POSTGRES_TEST_URL)

    # Generate 384-dim normalized vector
    emb1 = np.random.rand(384).astype(np.float32)
    emb1 /= np.linalg.norm(emb1)

    prompt = "What is Postgres pgvector?"
    response = "pgvector is an open-source vector similarity search extension for PostgreSQL."

    cache.write(prompt, emb1, response, ttl=300)
    assert cache.count() >= 1

    # Exact vector lookup
    hit = cache.lookup(emb1, threshold=0.90)
    assert hit is not None
    assert hit.prompt == prompt
    assert hit.response == response
    assert hit.similarity >= 0.99

    # Slightly perturbed vector (similarity ~0.95)
    emb_similar = emb1 + (np.random.rand(384).astype(np.float32) * 0.05)
    emb_similar /= np.linalg.norm(emb_similar)

    hit_sim = cache.lookup(emb_similar, threshold=0.85)
    assert hit_sim is not None
    assert hit_sim.prompt == prompt

    # Orthogonal/opposite vector (similarity < 0)
    emb_opposite = -emb1
    miss = cache.lookup(emb_opposite, threshold=0.80)
    assert miss is None


def test_postgres_metrics_store_full_suite():
    store = PostgresMetricsStore(POSTGRES_TEST_URL)
    trace_id = f"trace-pg-{int(time.time())}"
    span_id = f"span-pg-{int(time.time())}"

    entry = LogEntry(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=None,
        function_name="sql_query_pg",
        prompt="SELECT * FROM pgvector",
        response="Result rows",
        cache_hit=False,
        latency_ms=45.2,
        cost_estimate=0.00012,
        timestamp=time.time(),
        variant="primary",
    )
    store.log_request(entry)

    stats = store.get_stats()
    assert stats["total_calls"] >= 1
    assert stats["total_cost_usd"] >= 0.00012

    recent = store.get_recent_logs(limit=10)
    assert len(recent) >= 1
    assert any(r["trace_id"] == trace_id for r in recent)

    traces = store.get_traces(limit=10)
    assert any(t["trace_id"] == trace_id for t in traces)

    trends = store.get_trends(bucket_minutes=5, limit_buckets=10)
    assert isinstance(trends, list)

    canary = store.get_canary_stats()
    assert isinstance(canary, list)


def test_decorator_e2e_with_postgres(setup_postgres_env):
    cache = get_cache_backend()
    metrics = get_metrics_store()

    assert isinstance(cache, PostgresCacheBackend)
    assert isinstance(metrics, PostgresMetricsStore)

    call_count = {"n": 0}

    @track(cache=True, similarity_threshold=0.999)
    def my_pg_llm(prompt: str) -> str:
        call_count["n"] += 1
        return f"Postgres answer for {prompt}"

    prompt_str = f"Explain pgvector features {uuid.uuid4().hex}"

    # Call 1: Miss -> live call
    r1 = my_pg_llm(prompt_str)
    info1 = get_last_call_info()
    assert call_count["n"] == 1
    assert info1 is not None and not info1.cache_hit

    # Call 2: Hit -> cache hit via pgvector
    r2 = my_pg_llm(prompt_str)
    info2 = get_last_call_info()
    assert call_count["n"] == 1  # Function not called second time!
    assert r1 == r2
    assert info2 is not None and info2.cache_hit
