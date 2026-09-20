"""
lensllm.backends.postgres_backend
====================================
Postgres implementations for CacheBackend (pgvector) and MetricsStore.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional
import numpy as np

from lensllm.backends.base import CacheBackend, CacheEntry, MetricsStore

logger = logging.getLogger("lensllm")

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    psycopg2 = None
    RealDictCursor = None


class PostgresConnectionManager:
    """Thread-safe lazy singleton connection manager for Postgres."""

    def __init__(self, url: str):
        if not HAS_PSYCOPG2:
            raise ImportError(
                "psycopg2 is required for Postgres support. "
                "Install it via: pip install psycopg2-binary pgvector"
            )
        self._url = url
        self._conn: Optional[Any] = None
        self._lock = threading.Lock()
        self._initialized = False

    def get_conn(self) -> Any:
        with self._lock:
            if self._conn is None or self._conn.closed != 0:
                self._conn = psycopg2.connect(self._url)
                self._conn.autocommit = True
            if not self._initialized:
                self._init_tables(self._conn)
                self._initialized = True
            return self._conn

    def _init_tables(self, conn: Any) -> None:
        with conn.cursor() as cur:
            # Enable pgvector extension
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            except Exception as e:
                logger.warning(f"[lensllm] pgvector extension initialization warning: {e}")

            # Create cache_entries table with vector(384)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id SERIAL PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    embedding vector(384) NOT NULL,
                    response TEXT NOT NULL,
                    created_at DOUBLE PRECISION NOT NULL,
                    expires_at DOUBLE PRECISION,
                    hit_count INT DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at);
                """
            )

            # Try creating HNSW vector index for pgvector
            try:
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_cache_embedding_hnsw "
                    "ON cache_entries USING hnsw (embedding vector_cosine_ops);"
                )
            except Exception:
                pass  # Fall back to unindexed vector scan if HNSW fails

            # Create request_logs table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS request_logs (
                    id BIGSERIAL PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    span_id TEXT NOT NULL UNIQUE,
                    parent_span_id TEXT,
                    function_name TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT,
                    cache_hit INT NOT NULL DEFAULT 0,
                    latency_ms DOUBLE PRECISION,
                    cost_estimate DOUBLE PRECISION,
                    variant TEXT,
                    error TEXT,
                    timestamp DOUBLE PRECISION NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_req_logs_trace_id ON request_logs(trace_id);
                CREATE INDEX IF NOT EXISTS idx_req_logs_timestamp ON request_logs(timestamp);
                """
            )


class PostgresCacheBackend(CacheBackend):
    """Postgres + pgvector implementation of CacheBackend."""

    def __init__(self, url: str):
        self._manager = PostgresConnectionManager(url)

    def lookup(
        self,
        prompt_embedding: np.ndarray,
        threshold: float,
    ) -> Optional[CacheEntry]:
        try:
            conn = self._manager.get_conn()
            now = time.time()
            vec_str = "[" + ",".join(str(float(x)) for x in prompt_embedding) + "]"

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, prompt, response, created_at, hit_count,
                           (1 - (embedding <=> %s::vector)) AS similarity
                    FROM cache_entries
                    WHERE (expires_at IS NULL OR expires_at > %s)
                      AND (1 - (embedding <=> %s::vector)) >= %s
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT 1;
                    """,
                    (vec_str, now, vec_str, threshold, vec_str),
                )
                row = cur.fetchone()
                if not row:
                    return None

                entry_id, prompt, response, created_at, hit_count, similarity = row

                # Update hit count
                cur.execute(
                    "UPDATE cache_entries SET hit_count = hit_count + 1 WHERE id = %s;",
                    (entry_id,),
                )

                return CacheEntry(
                    id=entry_id,
                    prompt=prompt,
                    response=response,
                    similarity=float(similarity),
                    created_at=float(created_at),
                    hit_count=int(hit_count) + 1,
                )
        except Exception as e:
            logger.warning(f"[lensllm] PostgresCacheBackend lookup error: {e}")
            return None

    def write(
        self,
        prompt: str,
        prompt_embedding: np.ndarray,
        response: str,
        ttl: Optional[int] = None,
    ) -> None:
        try:
            conn = self._manager.get_conn()
            now = time.time()
            expires_at = (now + ttl) if ttl is not None else None
            vec_str = "[" + ",".join(str(float(x)) for x in prompt_embedding) + "]"

            with conn.cursor() as cur:
                # Cleanup expired
                cur.execute(
                    "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at <= %s;",
                    (now,),
                )
                cur.execute(
                    """
                    INSERT INTO cache_entries (prompt, embedding, response, created_at, expires_at, hit_count)
                    VALUES (%s, %s::vector, %s, %s, %s, 0);
                    """,
                    (prompt, vec_str, response, now, expires_at),
                )
        except Exception as e:
            logger.warning(f"[lensllm] PostgresCacheBackend write error: {e}")

    def count(self) -> int:
        try:
            conn = self._manager.get_conn()
            now = time.time()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM cache_entries WHERE expires_at IS NULL OR expires_at > %s;",
                    (now,),
                )
                row = cur.fetchone()
                return int(row[0]) if row else 0
        except Exception as e:
            logger.warning(f"[lensllm] PostgresCacheBackend count error: {e}")
            return 0


class PostgresMetricsStore(MetricsStore):
    """Postgres implementation of MetricsStore."""

    def __init__(self, url: str):
        self._manager = PostgresConnectionManager(url)

    def log_request(self, entry: Any) -> None:
        try:
            conn = self._manager.get_conn()
            response_text = entry.response or None
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO request_logs (
                        trace_id, span_id, parent_span_id, function_name,
                        prompt, response, cache_hit, latency_ms,
                        cost_estimate, variant, error, timestamp
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (span_id) DO NOTHING;
                    """,
                    (
                        entry.trace_id,
                        entry.span_id,
                        entry.parent_span_id,
                        entry.function_name,
                        entry.prompt,
                        response_text,
                        int(entry.cache_hit),
                        entry.latency_ms,
                        entry.cost_estimate,
                        entry.variant,
                        entry.error,
                        entry.timestamp,
                    ),
                )
        except Exception as e:
            logger.warning(f"[lensllm] PostgresMetricsStore log_request error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        try:
            conn = self._manager.get_conn()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
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
                row = cur.fetchone() or {}

                total = int(row.get("total_calls") or 0)
                hits = int(row.get("cache_hits") or 0)
                errors = int(row.get("error_count") or 0)

                return {
                    "total_calls": total,
                    "cache_hits": hits,
                    "cache_misses": int(row.get("cache_misses") or 0),
                    "hit_rate_pct": round((hits / total * 100) if total else 0, 1),
                    "avg_latency_ms": float(row.get("avg_latency_ms") or 0.0),
                    "total_cost_usd": float(row.get("total_cost_usd") or 0.0),
                    "error_count": errors,
                    "error_rate_pct": round((errors / total * 100) if total else 0, 1),
                }
        except Exception as e:
            logger.warning(f"[lensllm] PostgresMetricsStore get_stats error: {e}")
            return {
                "total_calls": 0, "cache_hits": 0, "cache_misses": 0,
                "hit_rate_pct": 0.0, "avg_latency_ms": 0.0, "total_cost_usd": 0.0,
                "error_count": 0, "error_rate_pct": 0.0,
            }

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            conn = self._manager.get_conn()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT trace_id, span_id, parent_span_id, function_name,
                           prompt, response, cache_hit, latency_ms,
                           cost_estimate, error, timestamp, variant
                    FROM request_logs
                    ORDER BY timestamp DESC
                    LIMIT %s;
                    """,
                    (limit,),
                )
                rows = cur.fetchall() or []
                return [dict(r) for r in rows]
        except Exception as e:
            logger.warning(f"[lensllm] PostgresMetricsStore get_recent_logs error: {e}")
            return []

    def get_trends(self, bucket_minutes: int = 5, limit_buckets: int = 24) -> List[Dict[str, Any]]:
        try:
            bucket_secs = bucket_minutes * 60
            since = time.time() - (bucket_secs * limit_buckets)
            conn = self._manager.get_conn()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        CAST((timestamp - %s) / %s AS INTEGER) AS bucket,
                        COUNT(*)                               AS total_calls,
                        COALESCE(SUM(cache_hit), 0)           AS cache_hits,
                        ROUND(AVG(CASE WHEN cache_hit=0 THEN latency_ms END)::numeric, 2)
                                                               AS avg_latency_ms
                    FROM request_logs
                    WHERE timestamp >= %s
                    GROUP BY bucket
                    ORDER BY bucket ASC;
                    """,
                    (since, bucket_secs, since),
                )
                rows = cur.fetchall() or []
                return [
                    {
                        "timestamp": since + r["bucket"] * bucket_secs,
                        "total_calls": int(r["total_calls"]),
                        "cache_hits": int(r["cache_hits"] or 0),
                        "avg_latency_ms": float(r["avg_latency_ms"] or 0.0),
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.warning(f"[lensllm] PostgresMetricsStore get_trends error: {e}")
            return []

    def get_cost_since(self, since_ts: float) -> float:
        try:
            conn = self._manager.get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COALESCE(SUM(cost_estimate), 0) FROM request_logs WHERE timestamp >= %s;",
                    (since_ts,),
                )
                row = cur.fetchone()
                return float(row[0]) if row else 0.0
        except Exception as e:
            logger.warning(f"[lensllm] PostgresMetricsStore get_cost_since error: {e}")
            return 0.0

    def get_canary_stats(self) -> List[Dict[str, Any]]:
        try:
            conn = self._manager.get_conn()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        COALESCE(variant, 'untagged')   AS variant,
                        COUNT(*)                         AS total_calls,
                        COALESCE(SUM(cache_hit), 0)      AS cache_hits,
                        ROUND(AVG(CASE WHEN cache_hit=0 THEN latency_ms END)::numeric, 2)
                                                         AS avg_latency_ms,
                        ROUND(COALESCE(SUM(cost_estimate), 0)::numeric, 6)
                                                         AS total_cost_usd,
                        COALESCE(SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END), 0)
                                                         AS error_count
                    FROM request_logs
                    GROUP BY COALESCE(variant, 'untagged')
                    ORDER BY variant;
                    """
                )
                rows = cur.fetchall() or []
                return [dict(r) for r in rows]
        except Exception as e:
            logger.warning(f"[lensllm] PostgresMetricsStore get_canary_stats error: {e}")
            return []

    def get_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            conn = self._manager.get_conn()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT trace_id, MIN(timestamp) AS first_ts
                    FROM request_logs
                    GROUP BY trace_id
                    ORDER BY first_ts DESC
                    LIMIT %s;
                    """,
                    (limit,),
                )
                trace_rows = cur.fetchall() or []

                traces = []
                for tr in trace_rows:
                    cur.execute(
                        """
                        SELECT trace_id, span_id, parent_span_id, function_name,
                               prompt, response, cache_hit, latency_ms,
                               cost_estimate, error, timestamp, variant
                        FROM request_logs
                        WHERE trace_id = %s
                        ORDER BY timestamp ASC;
                        """,
                        (tr["trace_id"],),
                    )
                    spans = [dict(s) for s in (cur.fetchall() or [])]
                    traces.append({
                        "trace_id": tr["trace_id"],
                        "span_count": len(spans),
                        "first_call_ts": float(tr["first_ts"]),
                        "functions": list({s["function_name"] for s in spans}),
                        "spans": spans,
                    })
                return traces
        except Exception as e:
            logger.warning(f"[lensllm] PostgresMetricsStore get_traces error: {e}")
            return []
