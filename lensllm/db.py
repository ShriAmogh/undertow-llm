"""
lensllm.db
==============
SQLite connection factory and schema management.

Design decisions:
    - WAL (Write-Ahead Logging) mode:
        SQLite's default journal mode serialises ALL readers while a write is in
        progress. WAL mode allows multiple concurrent readers alongside a single
        writer — critical here because the @track decorator writes a log row on
        every LLM call, while the FastAPI dashboard reads stats simultaneously.

    - check_same_thread=False:
        FastAPI/uvicorn uses a thread-pool for sync route handlers. We need the
        same connection (or a per-thread connection) to be usable from any worker
        thread. We use a per-call connection via get_connection() for simplicity
        at MVP scale.

Schema tables:
    cache_entries  — stores embeddings + responses for semantic cache
    request_logs   — one row per LLM call; primary observability store
"""

from __future__ import annotations

import sqlite3
from lensllm.config import get_config


# ── Connection factory ────────────────────────────────────────────────────────

def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """
    Open (or create) the SQLite database and return a connection with:
      - WAL journal mode enabled
      - Row factory set to sqlite3.Row (dict-like row access)
    """
    path = db_path or get_config().db_path
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")

    return conn


# ── Schema creation ───────────────────────────────────────────────────────────

_SCHEMA_SQL = """
-- Semantic cache: stores prompt embeddings and their cached responses
CREATE TABLE IF NOT EXISTS cache_entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt          TEXT    NOT NULL,
    -- Embedding stored as a BLOB (numpy array serialised with numpy.save)
    embedding       BLOB    NOT NULL,
    response        TEXT    NOT NULL,
    created_at      REAL    NOT NULL,   -- UNIX timestamp (float)
    expires_at      REAL,               -- NULL = no TTL
    hit_count       INTEGER NOT NULL DEFAULT 0
);

-- Primary observability log: one row per LLM call (hit OR miss)
CREATE TABLE IF NOT EXISTS request_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    -- trace_id groups multiple @track calls in a single logical chain
    trace_id        TEXT    NOT NULL,
    -- span_id identifies this specific call within the chain
    span_id         TEXT    NOT NULL UNIQUE,
    parent_span_id  TEXT,               -- NULL for root spans
    function_name   TEXT    NOT NULL,
    prompt          TEXT    NOT NULL,
    response        TEXT,               -- NULL if call failed
    cache_hit       INTEGER NOT NULL DEFAULT 0,  -- 0=miss, 1=hit
    latency_ms      REAL,               -- NULL on cache hit (0 ms, not measured)
    cost_estimate   REAL,               -- USD estimate; NULL if unknown
    variant         TEXT,               -- for A/B canary: 'primary' | 'canary'
    error           TEXT,               -- exception message if call failed
    timestamp       REAL    NOT NULL    -- UNIX timestamp of the call start
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_request_logs_trace_id  ON request_logs(trace_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_timestamp ON request_logs(timestamp);
"""


def init_db(db_path: str | None = None) -> None:
    """
    Create all tables and indexes if they do not already exist.
    Safe to call on every startup (uses IF NOT EXISTS throughout).
    """
    conn = get_connection(db_path)
    try:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
