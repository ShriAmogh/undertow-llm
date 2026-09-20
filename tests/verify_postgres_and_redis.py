"""
verify_postgres_and_redis.py
============================
Verification script to test and demonstrate simultaneous PostgreSQL (pgvector)
and Redis operations for undertow_llm.

Usage:
    python verify_postgres_and_redis.py
"""

import os
import sys
import time
import uuid
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Environment settings
POSTGRES_URL = (
    os.getenv("UNDERTOW_LLM_POSTGRES_URL")
    or os.getenv("POSTGRES_URL")
    or "postgresql://undertow_llm:undertow_llm@localhost:5432/gateway"
)
REDIS_URL = (
    os.getenv("UNDERTOW_LLM_REDIS_URL")
    or os.getenv("REDIS_URL")
    or "redis://localhost:6379"
)


def print_banner(title: str):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


def test_postgresql_connection():
    """Verify PostgreSQL connection and pgvector extension."""
    print_banner("1. PostgreSQL & pgvector Verification")

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        print("❌ psycopg2 module missing. Install via: pip install psycopg2-binary")
        return False

    print(f"🔌 Connecting to PostgreSQL at:\n   {POSTGRES_URL}")

    try:
        conn = psycopg2.connect(POSTGRES_URL)
        conn.autocommit = True
    except Exception as e:
        print(f"❌ PostgreSQL Connection Failed: {e}")
        return False

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Check pgvector extension
        cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
        ext = cur.fetchone()
        if ext:
            print(f"  ✅ pgvector Extension Installed: version {ext['extversion']}")
        else:
            print("  ⚠️ pgvector extension not found! Creating extension...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print("  ✅ pgvector extension enabled successfully.")

        # Test Postgres Cache Backend
        from undertow_llm.backends.postgres_backend import PostgresCacheBackend, PostgresMetricsStore

        cache_store = PostgresCacheBackend(POSTGRES_URL)
        metrics_store = PostgresMetricsStore(POSTGRES_URL)

        # Vector write test
        test_prompt = f"What is PostgreSQL pgvector integration? {uuid.uuid4().hex[:6]}"
        test_resp = "PostgreSQL + pgvector enables production vector similarity caching."
        emb = np.random.rand(384).astype(np.float32)
        emb /= np.linalg.norm(emb)

        cache_store.write(test_prompt, emb, test_resp, ttl=300)
        print("  ✅ Successfully inserted 384-dimensional vector embedding into Postgres cache_entries.")

        # Vector lookup test
        hit = cache_store.lookup(emb, threshold=0.90)
        if hit and hit.prompt == test_prompt:
            print(f"  ⚡ Vector Search Lookup Hit! Similarity: {hit.similarity:.4f}")
            print(f"     Prompt: '{hit.prompt[:45]}...'")
            print(f"     Response: '{hit.response}'")
        else:
            print("  ❌ Vector Lookup Failed")
            return False

        # Metrics log test
        stats = metrics_store.get_stats()
        print(f"  📊 Current Postgres Metrics Stats:")
        print(f"     Total Requests: {stats['total_calls']}")
        print(f"     Cache Hits:     {stats['cache_hits']} (⚡ {stats['hit_rate_pct']}%)")
        print(f"     Avg Latency:    {stats['avg_latency_ms']} ms")
        print(f"     Est. Cost:      ${stats['total_cost_usd']}")

    conn.close()
    return True


def test_redis_connection():
    """Verify Redis connection and key/counter operations."""
    print_banner("2. Redis Connection & Key Operations")

    try:
        import redis
    except ImportError:
        print("❌ redis module missing. Install via: pip install redis")
        return False

    print(f"🔌 Connecting to Redis at:\n   {REDIS_URL}")

    try:
        r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        pong = r.ping()
        if pong:
            print(f"  ✅ Redis Server PING Response: PONG ({r.info('server').get('redis_version')})")
    except Exception as e:
        print(f"❌ Redis Connection Failed: {e}")
        return False

    # Test rate limiter counter key in Redis
    test_key = f"undertow-llm:test_counter:{uuid.uuid4().hex[:6]}"
    r.set(test_key, 10, ex=60)
    val = r.get(test_key)
    print(f"  ✅ Redis Set/Get Verification: Key '{test_key}' = {val}")

    r.incrby(test_key, 5)
    new_val = r.get(test_key)
    print(f"  ✅ Redis Atomic Increment: New value = {new_val}")

    r.delete(test_key)
    return True


def test_e2e_decorator_integration():
    """Verify @track() decorator with live PostgreSQL and Redis backends."""
    print_banner("3. End-to-End @track Decorator Integration (Postgres + Redis)")

    from undertow_llm.config import configure, get_config
    from undertow_llm.backends.factory import get_cache_backend, get_metrics_store, reset_backends
    from undertow_llm.decorator import track, get_last_call_info

    reset_backends()
    configure(
        postgres_url=POSTGRES_URL,
        redis_url=REDIS_URL,
        cache_enabled=True,
        similarity_threshold=0.90,
    )

    cache = get_cache_backend()
    metrics = get_metrics_store()
    cfg = get_config()

    print(f"  Active Cache Backend:   {type(cache).__name__}")
    print(f"  Active Metrics Store:   {type(metrics).__name__}")
    print(f"  Active Redis URL:       {cfg.redis_url}")

    call_count = {"n": 0}

    @track(cache=True, similarity_threshold=0.95, rate_limit_rate=10.0)
    def call_llm(prompt: str) -> str:
        call_count["n"] += 1
        time.sleep(0.05)
        return f"LLM answer for: '{prompt}'"

    unique_prompt = f"Explain deep neural networks {uuid.uuid4().hex[:6]}"

    # Call 1: Miss -> live call -> written to Postgres
    print("\n  👉 Executing Call 1 (Expected: MISS -> Live call)...")
    res1 = call_llm(unique_prompt)
    info1 = get_last_call_info()
    print(f"     Response: '{res1}'")
    print(f"     Call Info: Cache Hit = {info1.cache_hit if info1 else False} | Latency = {info1.latency_ms if info1 else 0} ms")

    # Call 2: Hit -> cached in Postgres pgvector
    print("\n  👉 Executing Call 2 (Expected: HIT -> Retrieved from Postgres pgvector)...")
    res2 = call_llm(unique_prompt)
    info2 = get_last_call_info()
    print(f"     Response: '{res2}'")
    print(f"     Call Info: Cache Hit = {info2.cache_hit if info2 else False} | Latency = {info2.latency_ms if info2 else 0} ms")

    assert call_count["n"] == 1, "Function body should only be executed once due to Postgres caching!"
    assert info2 is not None and info2.cache_hit, "Call 2 should be a cache hit!"

    print("\n  ✅ End-to-End Decorator execution succeeded using PostgreSQL + Redis!")
    return True


def main():
    print(f"{'═' * 70}")
    print("  UNDERTOW_LLM DUAL BACKEND VERIFICATION (POSTGRESQL + REDIS)")
    print(f"{'═' * 70}")

    pg_ok = test_postgresql_connection()
    redis_ok = test_redis_connection()
    e2e_ok = test_e2e_decorator_integration()

    print_banner("Verification Summary")
    print(f"  PostgreSQL + pgvector: {'✅ ACTIVE & OPERATIONAL' if pg_ok else '❌ FAILED'}")
    print(f"  Redis Backend:         {'✅ ACTIVE & OPERATIONAL' if redis_ok else '❌ FAILED'}")
    print(f"  Decorator E2E Flow:    {'✅ PASSED' if e2e_ok else '❌ FAILED'}")

    if pg_ok and redis_ok and e2e_ok:
        print(f"\n🎉 ALL VERIFICATIONS PASSED SUCCESSFULLY!\n")
        sys.exit(0)
    else:
        print(f"\n❌ VERIFICATION FAILED\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
