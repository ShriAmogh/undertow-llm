"""
undertow_llm.cli
===============
Command-line interface for undertow_llm.

Entry point registered in pyproject.toml:
    undertow_llm = undertow_llm.cli:main

Commands:
    undertow-llm serve [--port PORT] [--db PATH]
        Launch the local observability dashboard.

    undertow_llm init-db [--db PATH]
        Explicitly initialise (or re-initialise) the SQLite database.
        Useful for first-time setup or schema migrations.
"""

import click
from undertow_llm.db import init_db
from undertow_llm.config import get_config


@click.group()
def main():
    """undertow-llm — LLM Observability & Reliability SDK"""
    pass


@main.command()
@click.option("--port", default=None, type=int, help="Dashboard port (default: 8080)")
@click.option("--db", default=None, help="Path to SQLite database file")
def serve(port: int | None, db: str | None):
    """Launch the local observability dashboard."""
    import uvicorn

    cfg = get_config()
    effective_port = port or cfg.dashboard_port

    # Ensure DB schema is up to date before starting the server
    init_db(db)

    click.echo(f"🚀  undertow_llm dashboard starting at http://localhost:{effective_port}")
    click.echo("    Press Ctrl+C to stop.\n")

    uvicorn.run(
        "undertow_llm.server.app:app",
        host="0.0.0.0",
        port=effective_port,
        reload=False,
        log_level="warning",
    )


@main.command("init-db")
@click.option("--db", default=None, help="Path to SQLite database file")
def init_db_cmd(db: str | None):
    """Initialise (or re-initialise) the SQLite database schema."""
    init_db(db)
    cfg = get_config()
    db_path = db or cfg.db_path
    click.echo(f"✅  Database initialised: {db_path}")
