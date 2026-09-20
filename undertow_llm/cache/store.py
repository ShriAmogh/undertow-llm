"""
undertow_llm.cache.store
=======================
Cache read/write layer alias backed by SQLite.
Re-exports SQLiteCacheBackend as CacheStore for backwards compatibility.
"""

from undertow_llm.backends.sqlite_backend import SQLiteCacheBackend as CacheStore
from undertow_llm.backends.base import CacheEntry

__all__ = ["CacheStore", "CacheEntry"]
