"""
gateway_sdk.backends.base
=========================
Abstract interfaces for CacheBackend and MetricsStore.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class CacheEntry:
    id: int | str
    prompt: str
    response: str
    similarity: float   # similarity score that caused the hit
    created_at: float
    hit_count: int


class CacheBackend(ABC):
    """Abstract interface for semantic cache storage backends."""

    @abstractmethod
    def lookup(
        self,
        prompt_embedding: np.ndarray,
        threshold: float,
    ) -> Optional[CacheEntry]:
        """
        Return the cached response with similarity >= threshold,
        or None if no match is found.
        """
        ...

    @abstractmethod
    def write(
        self,
        prompt: str,
        prompt_embedding: np.ndarray,
        response: str,
        ttl: Optional[int] = None,
    ) -> None:
        """Store a new cache entry."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total valid non-expired entries."""
        ...


class MetricsStore(ABC):
    """Abstract interface for request logging and dashboard metrics stores."""

    @abstractmethod
    def log_request(self, entry: Any) -> None:
        """Persist a single request log entry."""
        ...

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Summary dashboard metrics."""
        ...

    @abstractmethod
    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Recent request log records."""
        ...

    @abstractmethod
    def get_trends(self, bucket_minutes: int = 5, limit_buckets: int = 24) -> List[Dict[str, Any]]:
        """Time-series trend data points."""
        ...

    @abstractmethod
    def get_cost_since(self, since_ts: float) -> float:
        """Total cost incurred (USD) since timestamp."""
        ...

    @abstractmethod
    def get_canary_stats(self) -> List[Dict[str, Any]]:
        """A/B variant metrics."""
        ...

    @abstractmethod
    def get_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Traces with grouped spans."""
        ...
