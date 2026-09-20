"""
tests/test_db.py
================
Phase 1 tests: database schema creation and connection health.
"""

import sqlite3
import pytest
from lensllm.db import init_db, get_connection


class TestDbSchema:
    def test_init_db_creates_tables(self, temp_db):
        """All schema tables must exist after init_db()."""
        conn = get_connection(temp_db)
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()

        assert "cache_entries" in tables
        assert "request_logs" in tables

    def test_wal_mode_enabled(self, temp_db):
        """WAL journal mode must be active for concurrent reads."""
        conn = get_connection(temp_db)
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        conn.close()
        assert mode == "wal"

    def test_foreign_keys_enabled(self, temp_db):
        """Foreign key enforcement must be on."""
        conn = get_connection(temp_db)
        fk_on = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        conn.close()
        assert fk_on == 1

    def test_init_db_idempotent(self, temp_db):
        """Calling init_db() twice must not raise errors (IF NOT EXISTS)."""
        init_db(temp_db)  # second call — should be a no-op
        conn = get_connection(temp_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        assert len(tables) >= 2
