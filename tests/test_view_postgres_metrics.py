"""
test_view_postgres_metrics.py
==============================
Test script to seed test request log metrics into PostgreSQL and view/query the
metrics tables (`request_logs` and `cache_entries`).

Usage:
    python test_view_postgres_metrics.py
    # or
    pytest test_view_postgres_metrics.py
"""

import os
import sys
import time
import random
import uuid
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

# Get PostgreSQL URL from environment or fallback to default Docker URL
POSTGRES_URL = (
    os.getenv("LENSLLM_POSTGRES_URL")
    or os.getenv("POSTGRES_URL")
    or "postgresql://lensllm:lensllm@localhost:5432/gateway"
)

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

from lensllm.backends.postgres_backend import PostgresMetricsStore, PostgresCacheBackend
from lensllm.logging.store import LogEntry


def print_section(title: str):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


def seed_test_metrics_data(url: str) -> Dict[str, Any]:
    """Seed synthetic LLM request logs and metrics into PostgreSQL using PostgresMetricsStore."""
    store = PostgresMetricsStore(url)
    test_run_id = f"test-{uuid.uuid4().hex[:6]}"
    now = time.time()

    sample_logs = [
        # (function_name, prompt, response, cache_hit, latency_ms, cost_estimate, variant, error)
        ("chat_gpt4", "Summarize article on AI", "AI summary response", False, 450.0, 0.0035, "primary", None),
        ("chat_gpt4", "Summarize article on AI", "AI summary response", True, 2.5, 0.0, "primary", None),
        ("recommend_item", "Product recommendation for user 101", "Recommended items: [A, B]", False, 180.0, 0.0012, "canary_v2", None),
        ("recommend_item", "Product recommendation for user 101", "Recommended items: [A, B]", True, 1.8, 0.0, "canary_v2", None),
        ("code_gen", "Write a python sorting function", "def quicksort(arr): ...", False, 850.0, 0.0080, "primary", None),
        ("code_gen", "Refactor sql query", "SELECT * FROM users JOIN orders ...", False, 0.0, 0.0, "primary", "TimeoutError: LLM endpoint unreachable"),
        ("translate_txt", "Translate to Spanish: Hello world", "Hola mundo", False, 120.0, 0.0005, "canary_v2", None),
        ("translate_txt", "Translate to Spanish: Hello world", "Hola mundo", True, 1.2, 0.0, "canary_v2", None),
    ]

    inserted_count = 0
    for idx, (fn, prompt, resp, hit, lat, cost, variant, err) in enumerate(sample_logs):
        span_id = f"span-{test_run_id}-{idx}-{random.randint(1000, 9999)}"
        entry = LogEntry(
            trace_id=f"trace-{test_run_id}-{idx}",
            span_id=span_id,
            parent_span_id=None,
            function_name=fn,
            prompt=prompt,
            response=resp,
            cache_hit=hit,
            latency_ms=lat,
            cost_estimate=cost,
            timestamp=now - (len(sample_logs) - idx) * 10,
            variant=variant,
            error=err
        )
        store.log_request(entry)
        inserted_count += 1

    return {"test_run_id": test_run_id, "inserted_count": inserted_count}


def view_postgres_metrics_tables(conn) -> Dict[str, Any]:
    """Query and view metrics tables from PostgreSQL."""
    summary_results = {}

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # 1. Overall Metrics Summary
        print_section("1. PostgreSQL Metrics Table Summary (request_logs)")
        cur.execute(
            """
            SELECT
                COUNT(*)                                            AS total_calls,
                COALESCE(SUM(cache_hit), 0)                        AS cache_hits,
                COUNT(*) - COALESCE(SUM(cache_hit), 0)             AS cache_misses,
                ROUND(AVG(CASE WHEN cache_hit=0 THEN latency_ms END)::numeric, 2) AS avg_miss_latency_ms,
                ROUND(COALESCE(SUM(cost_estimate), 0)::numeric, 6) AS total_cost_usd,
                COALESCE(SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END), 0) AS error_count
            FROM request_logs;
            """
        )
        stats = cur.fetchone() or {}
        summary_results["stats"] = stats

        total = int(stats.get("total_calls") or 0)
        hits = int(stats.get("cache_hits") or 0)
        misses = int(stats.get("cache_misses") or 0)
        hit_rate = round((hits / total * 100) if total else 0.0, 1)

        print(f"  📊 Total Logged Calls:   {total}")
        print(f"  ⚡ Cache Hits:          {hits} ({hit_rate}%)")
        print(f"  🌐 Cache Misses:        {misses}")
        print(f"  ⏱️  Avg Miss Latency:    {stats.get('avg_miss_latency_ms') or 0.0} ms")
        print(f"  💵 Total Est Cost:      ${stats.get('total_cost_usd') or 0.0}")
        print(f"  ⚠️ Error Count:         {stats.get('error_count') or 0}")

        # 2. Canary & Variant Distribution
        print_section("2. Metrics by Variant / Model Canary")
        cur.execute(
            """
            SELECT
                COALESCE(variant, 'untagged')                      AS variant,
                COUNT(*)                                            AS calls,
                COALESCE(SUM(cache_hit), 0)                        AS hits,
                ROUND(AVG(CASE WHEN cache_hit=0 THEN latency_ms END)::numeric, 2) AS avg_latency_ms,
                ROUND(COALESCE(SUM(cost_estimate), 0)::numeric, 6) AS total_cost
            FROM request_logs
            GROUP BY COALESCE(variant, 'untagged')
            ORDER BY variant;
            """
        )
        variant_rows = cur.fetchall()
        summary_results["variants"] = variant_rows

        print(f"  {'VARIANT':<15} {'CALLS':<8} {'HITS':<8} {'AVG LATENCY':<15} {'TOTAL COST'}")
        print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*15} {'-'*12}")
        for v in variant_rows:
            print(f"  {v['variant']:<15} {v['calls']:<8} {v['hits']:<8} {str(v['avg_latency_ms'] or 0) + ' ms':<15} ${v['total_cost']}")

        # 3. Recent Logs View
        print_section("3. Recent Request Logs (Top 10 Rows)")
        cur.execute(
            """
            SELECT id, trace_id, function_name, prompt, cache_hit, latency_ms,
                   COALESCE(variant, 'untagged') AS variant, error, timestamp
            FROM request_logs
            ORDER BY id DESC
            LIMIT 10;
            """
        )
        recent_logs = cur.fetchall()
        summary_results["recent_logs"] = recent_logs

        print(f"  {'ID':<5} {'FUNCTION':<15} {'VARIANT':<12} {'HIT':<6} {'LATENCY':<10} {'STATUS'}")
        print(f"  {'-'*5} {'-'*15} {'-'*12} {'-'*6} {'-'*10} {'-'*15}")
        for r in recent_logs:
            hit_str = "⚡ HIT" if r['cache_hit'] else "🌐 MISS"
            lat_str = f"{r['latency_ms']:,.1f} ms" if r['latency_ms'] is not None else "0 ms"
            status_str = "❌ ERROR" if r['error'] else "✅ OK"
            print(f"  {r['id']:<5} {r['function_name']:<15} {r['variant']:<12} {hit_str:<6} {lat_str:<10} {status_str}")

        # 4. Check Cache Entries table (pgvector)
        print_section("4. Semantic Cache Table (cache_entries)")
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'cache_entries'
            );
            """
        )
        has_cache_table = cur.fetchone()['exists']
        if has_cache_table:
            cur.execute(
                """
                SELECT id, prompt, hit_count, created_at, expires_at
                FROM cache_entries
                ORDER BY id DESC
                LIMIT 5;
                """
            )
            cache_rows = cur.fetchall()
            print(f"  Total Cache Rows Preview: {len(cache_rows)}")
            for c in cache_rows:
                p_preview = (c['prompt'][:40] + '...') if len(c['prompt']) > 40 else c['prompt']
                print(f"  - [ID {c['id']}] Hits: {c['hit_count']} | Prompt: '{p_preview}'")
        else:
            print("  (cache_entries table does not exist yet)")

    return summary_results


def run_test():
    """Main test execution function."""
    print("🔌 Connecting to PostgreSQL at:")
    print(f"   {POSTGRES_URL}")

    if not PSYCOPG2_AVAILABLE:
        print("❌ Error: psycopg2 module is not installed.")
        sys.exit(1)

    try:
        conn = psycopg2.connect(POSTGRES_URL)
    except Exception as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}")
        print("💡 Ensure PostgreSQL is running (e.g. docker compose up -d)")
        sys.exit(1)

    try:
        # Step 1: Seed test data via PostgresMetricsStore
        seed_res = seed_test_metrics_data(POSTGRES_URL)
        print(f"\n✅ Seeded {seed_res['inserted_count']} test request log entries into PostgreSQL.")

        # Step 2: View PostgreSQL metrics table
        metrics_res = view_postgres_metrics_tables(conn)

        # Step 3: Assertions to verify correctness
        stats = metrics_res["stats"]
        assert stats["total_calls"] > 0, "Expected total_calls to be > 0"
        assert len(metrics_res["recent_logs"]) > 0, "Expected recent_logs to have items"

        print(f"\n{'═' * 70}")
        print("✅ SUCCESS: PostgreSQL metrics table viewed and verified successfully!")
        print(f"{'═' * 70}\n")
    finally:
        conn.close()


def test_postgres_metrics_view():
    """Pytest entrypoint."""
    if not PSYCOPG2_AVAILABLE:
        import pytest
        pytest.skip("psycopg2 is not installed")
    try:
        conn = psycopg2.connect(POSTGRES_URL)
        conn.close()
    except Exception:
        import pytest
        pytest.skip("PostgreSQL container is not reachable")

    run_test()


if __name__ == "__main__":
    run_test()
