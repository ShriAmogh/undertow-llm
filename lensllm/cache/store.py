"""
lensllm.cache.store
=======================
Cache read/write layer alias backed by SQLite.
Re-exports SQLiteCacheBackend as CacheStore for backwards compatibility.
"""

from lensllm.backends.sqlite_backend import SQLiteCacheBackend as CacheStore
from lensllm.backends.base import CacheEntry

__all__ = ["CacheStore", "CacheEntry"]
