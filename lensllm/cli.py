"""
lensllm.cli
===============
Command-line interface for lensllm.

Entry point registered in pyproject.toml:
    lensllm = lensllm.cli:main

Commands:
    lensllm serve [--port PORT] [--db PATH]
        Launch the local observability dashboard.

    lensllm init-db [--db PATH]
        Explicitly initialise (or re-initialise) the SQLite database.
        Useful for first-time setup or schema migrations.
"""

import click
from lensllm.db import init_db
from lensllm.config import get_config


@click.group()
def main():
    """lensllm — LLM Observability & Reliability SDK"""
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

    click.echo(f"🚀  lensllm dashboard starting at http://localhost:{effective_port}")
    click.echo("    Press Ctrl+C to stop.\n")

    uvicorn.run(
        "lensllm.server.app:app",
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
