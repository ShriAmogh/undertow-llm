"""
lensllm.logging.store
=========================
Request log persistence and stats aggregation.
Re-exports SQLiteMetricsStore as LogStore for backwards compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lensllm.backends.sqlite_backend import SQLiteMetricsStore as LogStore


@dataclass
class LogEntry:
    trace_id: str
    span_id: str
    function_name: str
    prompt: str
    response: Optional[str]
    cache_hit: bool
    latency_ms: Optional[float]
    cost_estimate: Optional[float]
    timestamp: float
    error: Optional[str] = None
    parent_span_id: Optional[str] = None
    variant: Optional[str] = None


__all__ = ["LogStore", "LogEntry"]
