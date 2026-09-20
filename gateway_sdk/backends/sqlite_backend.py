"""
gateway_sdk.backends.sqlite_backend
===================================
SQLite implementations for CacheBackend and MetricsStore.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import numpy as np

from gateway_sdk.backends.base import CacheBackend, CacheEntry, MetricsStore
from gateway_sdk.db import get_connection, init_db
from gateway_sdk.cache.semantic import (
    cosine_similarity,
    bytes_to_embedding,
    embedding_to_bytes,
)


class SQLiteCacheBackend(CacheBackend):
    """SQLite implementation of CacheBackend (linear embedding scan)."""

    def __init__(self, db_path: str | None = None):
        from gateway_sdk.config import get_config
        self._db_path = db_path or get_config().db_path
        init_db(self._db_path)

    def lookup(
        self,
        prompt_embedding: np.ndarray,
        threshold: float,
    ) -> Optional[CacheEntry]:
        now = time.time()
        conn = get_connection(self._db_path)
        try:
            rows = conn.execute(
                """
                SELECT id, prompt, embedding, response, created_at, hit_count
                FROM cache_entries
                WHERE expires_at IS NULL OR expires_at > ?
                """,
                (now,),
            ).fetchall()

            best_entry = None
            best_score = -1.0

            for row in rows:
                cached_emb = bytes_to_embedding(row["embedding"])
                score = cosine_similarity(prompt_embedding, cached_emb)
                if score >= threshold and score > best_score:
                    best_score = score
                    best_entry = row

            if best_entry is None:
                return None

            conn.execute(
                "UPDATE cache_entries SET hit_count = hit_count + 1 WHERE id = ?",
                (best_entry["id"],),
            )
            conn.commit()

            return CacheEntry(
                id=best_entry["id"],
                prompt=best_entry["prompt"],
                response=best_entry["response"],
                similarity=best_score,
                created_at=best_entry["created_at"],
                hit_count=best_entry["hit_count"] + 1,
            )
        finally:
            conn.close()

    def write(
        self,
        prompt: str,
        prompt_embedding: np.ndarray,
        response: str,
        ttl: Optional[int] = None,
    ) -> None:
        now = time.time()
        expires_at = (now + ttl) if ttl is not None else None
        emb_blob = embedding_to_bytes(prompt_embedding)

        conn = get_connection(self._db_path)
        try:
            conn.execute(
                "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            conn.execute(
                """
                INSERT INTO cache_entries (prompt, embedding, response, created_at, expires_at, hit_count)
                VALUES (?, ?, ?, ?, ?, 0)
                """,
                (prompt, emb_blob, response, now, expires_at),
            )
            conn.commit()
        finally:
            conn.close()

    def count(self) -> int:
        now = time.time()
        conn = get_connection(self._db_path)
        try:
            return conn.execute(
                "SELECT COUNT(*) FROM cache_entries WHERE expires_at IS NULL OR expires_at > ?",
                (now,),
            ).fetchone()[0]
        finally:
            conn.close()


class SQLiteMetricsStore(MetricsStore):
    """SQLite implementation of MetricsStore."""

    def __init__(self, db_path: str | None = None):
        from gateway_sdk.config import get_config
        self._db_path = db_path or get_config().db_path
        init_db(self._db_path)

    def log_request(self, entry: Any) -> None:
        response_text = entry.response or None
        conn = get_connection(self._db_path)
        try:
            conn.execute(
                """
                INSERT INTO request_logs (
                    trace_id, span_id, parent_span_id, function_name,
                    prompt, response, cache_hit, latency_ms,
                    cost_estimate, variant, error, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            conn.commit()
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        conn = get_connection(self._db_path)
        try:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)                                            AS total_calls,
                    SUM(cache_hit)                                      AS cache_hits,
                    COUNT(*) - SUM(cache_hit)                          AS cache_misses,
                    ROUND(AVG(CASE WHEN cache_hit=0 THEN latency_ms END), 2)
                                                                        AS avg_latency_ms,
                    ROUND(COALESCE(SUM(cost_estimate), 0), 6)          AS total_cost_usd,
                    SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS error_count
                FROM request_logs
                """
            ).fetchone()

            total = row["total_calls"] or 0
            hits = row["cache_hits"] or 0
            errors = row["error_count"] or 0

            return {
                "total_calls": total,
                "cache_hits": hits,
                "cache_misses": row["cache_misses"] or 0,
                "hit_rate_pct": round((hits / total * 100) if total else 0, 1),
                "avg_latency_ms": row["avg_latency_ms"] or 0.0,
                "total_cost_usd": row["total_cost_usd"] or 0.0,
                "error_count": errors,
                "error_rate_pct": round((errors / total * 100) if total else 0, 1),
            }
        finally:
            conn.close()

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        conn = get_connection(self._db_path)
        try:
            rows = conn.execute(
                """
                SELECT trace_id, span_id, parent_span_id, function_name,
                       prompt, response, cache_hit, latency_ms,
                       cost_estimate, error, timestamp, variant
                FROM request_logs
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_trends(self, bucket_minutes: int = 5, limit_buckets: int = 24) -> List[Dict[str, Any]]:
        bucket_secs = bucket_minutes * 60
        since = time.time() - (bucket_secs * limit_buckets)

        conn = get_connection(self._db_path)
        try:
            rows = conn.execute(
                """
                SELECT
                    CAST((timestamp - ?) / ? AS INTEGER) AS bucket,
                    COUNT(*)                              AS total_calls,
                    SUM(cache_hit)                        AS cache_hits,
                    ROUND(AVG(CASE WHEN cache_hit=0 THEN latency_ms END), 2)
                                                          AS avg_latency_ms
                FROM request_logs
                WHERE timestamp >= ?
                GROUP BY bucket
                ORDER BY bucket ASC
                """,
                (since, bucket_secs, since),
            ).fetchall()

            return [
                {
                    "timestamp": since + row["bucket"] * bucket_secs,
                    "total_calls": row["total_calls"],
                    "cache_hits": row["cache_hits"] or 0,
                    "avg_latency_ms": row["avg_latency_ms"] or 0.0,
                }
                for row in rows
            ]
        finally:
            conn.close()

    def get_cost_since(self, since_ts: float) -> float:
        conn = get_connection(self._db_path)
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_estimate), 0) FROM request_logs WHERE timestamp >= ?",
                (since_ts,),
            ).fetchone()
            return float(row[0])
        finally:
            conn.close()

    def get_canary_stats(self) -> List[Dict[str, Any]]:
        conn = get_connection(self._db_path)
        try:
            rows = conn.execute(
                """
                SELECT
                    COALESCE(variant, 'untagged')   AS variant,
                    COUNT(*)                         AS total_calls,
                    SUM(cache_hit)                   AS cache_hits,
                    ROUND(AVG(CASE WHEN cache_hit=0 THEN latency_ms END), 2)
                                                     AS avg_latency_ms,
                    ROUND(COALESCE(SUM(cost_estimate), 0), 6)
                                                     AS total_cost_usd,
                    SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END)
                                                     AS error_count
                FROM request_logs
                GROUP BY COALESCE(variant, 'untagged')
                ORDER BY variant
                """
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        conn = get_connection(self._db_path)
        try:
            trace_rows = conn.execute(
                """
                SELECT trace_id, MIN(timestamp) AS first_ts
                FROM request_logs
                GROUP BY trace_id
                ORDER BY first_ts DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            traces = []
            for tr in trace_rows:
                spans = conn.execute(
                    """
                    SELECT trace_id, span_id, parent_span_id, function_name,
                           prompt, response, cache_hit, latency_ms,
                           cost_estimate, error, timestamp, variant
                    FROM request_logs
                    WHERE trace_id = ?
                    ORDER BY timestamp ASC
                    """,
                    (tr["trace_id"],),
                ).fetchall()
                span_list = [dict(s) for s in spans]
                traces.append({
                    "trace_id": tr["trace_id"],
                    "span_count": len(span_list),
                    "first_call_ts": tr["first_ts"],
                    "functions": list({s["function_name"] for s in span_list}),
                    "spans": span_list,
                })
            return traces
        finally:
            conn.close()
