"""
lensllm.config
==================
Global SDK configuration using a dataclass.

Usage:
    from lensllm import configure

    configure(
        db_path="my_app.db",
        cache_enabled=True,
        similarity_threshold=0.92,
        rate_limit_rate=2.0,
    )

All @track() decorator parameters can also be set per-call; per-call values
always take precedence over the global config.

Technical note:
    We use a module-level singleton (_config) rather than a thread-local or
    context-var because the config is write-once at startup and then read-only
    during request handling — no concurrency concern.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()  # load .env file if present


@dataclass
class GatewayConfig:
    # ── Storage ──────────────────────────────────────────────────────────────
    db_path: str = field(
        default_factory=lambda: os.getenv("LENSLLM_DB_PATH", "lensllm.db")
    )
    postgres_url: str | None = field(
        default_factory=lambda: os.getenv("LENSLLM_POSTGRES_URL") or os.getenv("POSTGRES_URL")
    )
    redis_url: str | None = field(
        default_factory=lambda: os.getenv("LENSLLM_REDIS_URL") or os.getenv("REDIS_URL")
    )

    # ── Semantic Cache ────────────────────────────────────────────────────────
    cache_enabled: bool = True
    similarity_threshold: float = 0.92   # cosine similarity [0, 1]
    cache_ttl: int | None = None          # seconds; None = no expiry

    # ── Rate Limiter (token-bucket) ───────────────────────────────────────────
    # rate = tokens added per second; max_tokens = bucket capacity (burst limit)
    rate_limit_rate: float = 2.0          # tokens/sec
    rate_limit_max_tokens: float = 10.0   # burst limit

    # ── Retry ─────────────────────────────────────────────────────────────────
    retries: int = 3
    base_delay: float = 1.0               # seconds
    max_delay: float = 60.0               # seconds
    jitter: bool = True

    # ── Dashboard ─────────────────────────────────────────────────────────────
    dashboard_port: int = field(
        default_factory=lambda: int(os.getenv("LENSLLM_DASHBOARD_PORT", "8080"))
    )


# ── Module-level singleton ────────────────────────────────────────────────────
_config = GatewayConfig()


def configure(**kwargs) -> GatewayConfig:
    """
    Update global SDK configuration.

    Example:
        configure(db_path="prod.db", rate_limit_rate=5.0, retries=5)

    Returns the updated GatewayConfig so callers can inspect it.
    """
    global _config
    for key, value in kwargs.items():
        if not hasattr(_config, key):
            raise ValueError(
                f"Unknown lensllm config key: '{key}'. "
                f"Valid keys: {list(_config.__dataclass_fields__.keys())}"
            )
        setattr(_config, key, value)

    from lensllm.backends.factory import reset_backends
    reset_backends()

    # Pre-warm the embedding model in a background thread so the first @track
    # call doesn't block for ~8-10 s loading model weights from disk.
    if _config.cache_enabled:
        from lensllm.cache.semantic import prewarm
        prewarm()

    return _config


def get_config() -> GatewayConfig:
    """Return the current global configuration (read-only use)."""
    return _config
