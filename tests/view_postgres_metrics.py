"""
view_postgres_metrics.py
========================
Inspect metrics and semantic cache stored in PostgreSQL (with pgvector).

Run:
    python view_postgres_metrics.py
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("❌ psycopg2 is required. Install via: pip install psycopg2-binary")
    sys.exit(1)


POSTGRES_URL = (
    os.getenv("LENSLLM_POSTGRES_URL")
    or os.getenv("POSTGRES_URL")
    or "postgresql://lensllm:lensllm@localhost:5432/gateway"
)


def print_banner(title: str):
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")


def main():
    print(f"🔌 Connecting to PostgreSQL at:\n   {POSTGRES_URL}")

    try:
        conn = psycopg2.connect(POSTGRES_URL)
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        print("💡 Make sure Postgres container is running: docker compose up -d")
        sys.exit(1)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:

        # ── 1. Summary Statistics ──────────────────────────────────────────────
        print_banner("1. Summary Observability Statistics")
        cur.execute(
            """
            SELECT
                COUNT(*)                                            AS total_calls,
                COALESCE(SUM(cache_hit), 0)                        AS cache_hits,
                COUNT(*) - COALESCE(SUM(cache_hit), 0)             AS cache_misses,
                ROUND(AVG(CASE WHEN cache_hit=0 THEN latency_ms END)::numeric, 2)
                                                                    AS avg_latency_ms,
                ROUND(COALESCE(SUM(cost_estimate), 0)::numeric, 6) AS total_cost_usd,
                COALESCE(SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END), 0) AS error_count
            FROM request_logs;
            """
        )
        stats = cur.fetchone() or {}

        total = int(stats.get("total_calls") or 0)
        hits = int(stats.get("cache_hits") or 0)
        misses = int(stats.get("cache_misses") or 0)
        hit_rate = round((hits / total * 100) if total else 0.0, 1)

        print(f"  Total Requests:  {total}")
        print(f"  Cache Hits:      {hits}  (⚡ {hit_rate}%)")
        print(f"  Cache Misses:    {misses}")
        print(f"  Avg LLM Latency: {stats.get('avg_latency_ms') or 0.0} ms")
        print(f"  Est. Total Cost: ${stats.get('total_cost_usd') or 0.0}")
        print(f"  Error Count:     {stats.get('error_count') or 0}")

        # ── 2. Semantic Cache Table ─────────────────────────────────────────────
        print_banner("2. Semantic Cache Entries (pgvector)")
        cur.execute(
            """
            SELECT id, prompt, response, created_at, expires_at, hit_count,
                   vector_dims(embedding) as dim
            FROM cache_entries
            ORDER BY id DESC
            LIMIT 10;
            """
        )
        cache_rows = cur.fetchall()
        print(f"  Total Cache Rows: {len(cache_rows)}\n")

        if cache_rows:
            print(f"  {'ID':<5} {'DIMS':<6} {'HITS':<5} {'PROMPT':<35} {'RESPONSE PREVIEW'}")
            print(f"  {'-'*5} {'-'*6} {'-'*5} {'-'*35} {'-'*30}")
            for r in cache_rows:
                prompt_short = (r['prompt'][:32] + '...') if len(r['prompt']) > 32 else r['prompt']
                resp_short = (r['response'][:30].replace('\n', ' ') + '...') if len(r['response']) > 30 else r['response'].replace('\n', ' ')
                print(f"  {r['id']:<5} {r['dim']:<6} {r['hit_count']:<5} {prompt_short:<35} {resp_short}")
        else:
            print("  (No cache entries found)")

        # ── 3. Recent Request Logs ─────────────────────────────────────────────
        print_banner("3. Recent Request Logs (Last 10 Rows)")
        cur.execute(
            """
            SELECT id, trace_id, function_name, prompt, cache_hit, latency_ms,
                   COALESCE(variant, 'untagged') as variant, timestamp
            FROM request_logs
            ORDER BY timestamp DESC
            LIMIT 10;
            """
        )
        log_rows = cur.fetchall()

        if log_rows:
            print(f"  {'TIME':<10} {'FUNCTION':<15} {'HIT':<6} {'LATENCY':<10} {'VARIANT':<10} {'PROMPT'}")
            print(f"  {'-'*10} {'-'*15} {'-'*6} {'-'*10} {'-'*10} {'-'*30}")
            for r in log_rows:
                t_str = time.strftime('%H:%M:%S', time.localtime(r['timestamp']))
                hit_str = "⚡ HIT" if r['cache_hit'] else "🌐 MISS"
                lat_str = f"{r['latency_ms']:,.0f} ms" if r['latency_ms'] is not None else "0 ms"
                p_short = (r['prompt'][:30] + '...') if len(r['prompt']) > 30 else r['prompt']
                print(f"  {t_str:<10} {r['function_name']:<15} {hit_str:<6} {lat_str:<10} {r['variant']:<10} {p_short}")
        else:
            print("  (No request logs found)")

        # ── 4. Canary A/B Stats ───────────────────────────────────────────────
        print_banner("4. Canary A/B Variant Aggregation")
        cur.execute(
            """
            SELECT
                COALESCE(variant, 'untagged')   AS variant,
                COUNT(*)                         AS total_calls,
                COALESCE(SUM(cache_hit), 0)      AS cache_hits,
                ROUND(AVG(CASE WHEN cache_hit=0 THEN latency_ms END)::numeric, 2)
                                                 AS avg_latency_ms,
                ROUND(COALESCE(SUM(cost_estimate), 0)::numeric, 6)
                                                 AS total_cost_usd
            FROM request_logs
            GROUP BY COALESCE(variant, 'untagged')
            ORDER BY variant;
            """
        )
        canary_rows = cur.fetchall()

        for c in canary_rows:
            print(f"  Variant: {c['variant']:<10} | Calls: {c['total_calls']:<4} | Hits: {c['cache_hits']:<4} | Avg Latency: {c['avg_latency_ms']} ms | Cost: ${c['total_cost_usd']}")

    conn.close()
    print("\n✅ Verification finished cleanly.\n")


if __name__ == "__main__":
    main()
