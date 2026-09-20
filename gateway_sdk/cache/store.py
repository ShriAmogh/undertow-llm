"""
gateway_sdk.cache.store
=======================
Cache read/write layer alias backed by SQLite.
Re-exports SQLiteCacheBackend as CacheStore for backwards compatibility.
"""

from gateway_sdk.backends.sqlite_backend import SQLiteCacheBackend as CacheStore
from gateway_sdk.backends.base import CacheEntry

__all__ = ["CacheStore", "CacheEntry"]
