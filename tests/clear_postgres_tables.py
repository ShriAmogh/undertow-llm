"""
clear_postgres_tables.py
========================
Utility script to clear all cache entries and request logs from PostgreSQL.

Usage:
    python tests/clear_postgres_tables.py
"""

from __future__ import annotations

import os
import sys
from dotenv import load_dotenv

load_dotenv()

try:
    import psycopg2
except ImportError:
    print("❌ psycopg2 is required. Install via: pip install psycopg2-binary")
    sys.exit(1)

POSTGRES_URL = (
    os.getenv("UNDERTOW_LLM_POSTGRES_URL")
    or os.getenv("POSTGRES_URL")
    or "postgresql://undertow_llm:undertow_llm@localhost:5432/gateway"
)


def clear_postgres_tables(postgres_url: str = POSTGRES_URL) -> None:
    """Connect to PostgreSQL and truncate cache_entries and request_logs tables."""
    print(f"🔌 Connecting to PostgreSQL at:\n   {postgres_url}")

    try:
        conn = psycopg2.connect(postgres_url)
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        print("💡 Make sure PostgreSQL is running (e.g. docker compose up -d)")
        sys.exit(1)

    try:
        with conn.cursor() as cur:
            # Check existing row counts before clearing
            cur.execute("SELECT COUNT(*) FROM cache_entries;")
            cache_count_before = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM request_logs;")
            logs_count_before = cur.fetchone()[0]

            print(
                f"📊 Current row count: {cache_count_before} cache entries, {logs_count_before} request logs"
            )

            # Truncate tables and restart primary key identity sequences
            cur.execute("TRUNCATE TABLE cache_entries, request_logs RESTART IDENTITY CASCADE;")
            conn.commit()

            print("🧹 Cleared tables 'cache_entries' and 'request_logs' successfully.")
            print("✅ Sequences reset (RESTART IDENTITY CASCADE).")
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Error truncating PostgreSQL tables: {e}")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    clear_postgres_tables()
