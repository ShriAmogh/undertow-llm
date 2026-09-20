"""
gateway_sdk.backends.factory
============================
Singleton factory for CacheBackend and MetricsStore instances.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from gateway_sdk.backends.base import CacheBackend, MetricsStore

logger = logging.getLogger("gateway_sdk")

_cache_backend: Optional[CacheBackend] = None
_metrics_store: Optional[MetricsStore] = None


def get_active_backend_label() -> str:
    """Return human-readable label for active backend configuration."""
    from gateway_sdk.config import get_config
    cfg = get_config()

    pg_active = False
    pg_url = get_postgres_url()
    if pg_url:
        try:
            backend = get_cache_backend()
            from gateway_sdk.backends.postgres_backend import PostgresCacheBackend
            if isinstance(backend, PostgresCacheBackend):
                pg_active = True
        except Exception:
            pg_active = False

    redis_url = cfg.redis_url or os.getenv("GATEWAY_SDK_REDIS_URL") or os.getenv("REDIS_URL")
    redis_active = bool(redis_url and redis_url.strip())

    if pg_active:
        if redis_active:
            return "PostgreSQL + Redis"
        return "PostgreSQL"
    else:
        if redis_active:
            return "SQLite + Redis"
        return "SQLite"


def get_postgres_url() -> Optional[str]:
    """Return configured Postgres URL if present."""
    from gateway_sdk.config import get_config
    cfg = get_config()
    if cfg.postgres_url is None:
        return None
    env_url = os.getenv("GATEWAY_SDK_POSTGRES_URL") or os.getenv("POSTGRES_URL")
    if env_url == "":
        return None
    return env_url or cfg.postgres_url


def get_cache_backend(db_path: Optional[str] = None) -> CacheBackend:
    """
    Return CacheBackend instance.

    If GATEWAY_SDK_POSTGRES_URL or POSTGRES_URL is set and reachable, returns PostgresCacheBackend.
    Otherwise returns SQLiteCacheBackend (bound to db_path if provided).
    """
    global _cache_backend
    pg_url = get_postgres_url()
    if pg_url:
        if _cache_backend is None or not hasattr(_cache_backend, "_manager"):
            try:
                from gateway_sdk.backends.postgres_backend import PostgresCacheBackend
                backend = PostgresCacheBackend(pg_url)
                backend._manager.get_conn()  # probe connection
                _cache_backend = backend
                logger.info("[gateway-sdk] Using PostgresCacheBackend (pgvector)")
            except Exception as e:
                logger.warning(
                    f"[gateway-sdk] Failed to connect to Postgres ({e}), falling back to SQLite."
                )
                from gateway_sdk.backends.sqlite_backend import SQLiteCacheBackend
                _cache_backend = SQLiteCacheBackend(db_path=db_path)
        return _cache_backend

    if db_path is not None:
        from gateway_sdk.backends.sqlite_backend import SQLiteCacheBackend
        return SQLiteCacheBackend(db_path=db_path)

    if _cache_backend is None:
        from gateway_sdk.backends.sqlite_backend import SQLiteCacheBackend
        _cache_backend = SQLiteCacheBackend()
    return _cache_backend


def get_metrics_store(db_path: Optional[str] = None) -> MetricsStore:
    """
    Return MetricsStore instance.

    If GATEWAY_SDK_POSTGRES_URL or POSTGRES_URL is set and reachable, returns PostgresMetricsStore.
    Otherwise returns SQLiteMetricsStore (bound to db_path if provided).
    """
    global _metrics_store
    pg_url = get_postgres_url()
    if pg_url:
        if _metrics_store is None or not hasattr(_metrics_store, "_manager"):
            try:
                from gateway_sdk.backends.postgres_backend import PostgresMetricsStore
                store = PostgresMetricsStore(pg_url)
                store._manager.get_conn()  # probe connection
                _metrics_store = store
                logger.info("[gateway-sdk] Using PostgresMetricsStore")
            except Exception as e:
                logger.warning(
                    f"[gateway-sdk] Failed to connect to Postgres ({e}), falling back to SQLite."
                )
                from gateway_sdk.backends.sqlite_backend import SQLiteMetricsStore
                _metrics_store = SQLiteMetricsStore(db_path=db_path)
        return _metrics_store

    if db_path is not None:
        from gateway_sdk.backends.sqlite_backend import SQLiteMetricsStore
        return SQLiteMetricsStore(db_path=db_path)

    if _metrics_store is None:
        from gateway_sdk.backends.sqlite_backend import SQLiteMetricsStore
        _metrics_store = SQLiteMetricsStore()
    return _metrics_store


def reset_backends() -> None:
    """Reset singleton instances (used in testing)."""
    global _cache_backend, _metrics_store
    _cache_backend = None
    _metrics_store = None
