"""
lensllm.server.app
======================
FastAPI dashboard backend — Phase 2 full implementation.

Endpoints:
    GET /              → dashboard HTML (Jinja2 + Chart.js)
    GET /api/stats     → aggregated metrics JSON
    GET /api/logs      → recent request log (paginated)
    GET /api/trends    → time-series data for Chart.js charts
    GET /health        → simple liveness check
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from lensllm.db import init_db
from lensllm.backends.factory import (
    get_cache_backend,
    get_metrics_store,
    get_active_backend_label,
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="lensllm Dashboard",
    description="LLM Observability & Reliability SDK",
    version="0.1.0",
    docs_url="/api/docs",
)

# ── Static files & templates ──────────────────────────────────────────────────
_HERE = Path(__file__).parent
templates = Jinja2Templates(directory=str(_HERE / "templates"))
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")


@app.on_event("startup")
async def on_startup():
    """Ensure DB schema is present when the server starts."""
    init_db()


# ── HTML Dashboard ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the main dashboard page."""
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"backend_label": get_active_backend_label()},
    )


from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

# ── Favicon Route ─────────────────────────────────────────────────────────────
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    fav_path = _HERE / "static" / "favicon.png"
    if fav_path.exists():
        return FileResponse(fav_path, media_type="image/png")
    return JSONResponse({"error": "not found"}, status_code=404)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


from decimal import Decimal
from typing import Any


def _to_json_friendly(obj: Any) -> Any:
    """Recursively convert Decimal instances to floats for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_json_friendly(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_friendly(item) for item in obj]
    return obj


def safe_json_response(content: Any, status_code: int = 200) -> JSONResponse:
    """Return a JSONResponse with all Decimal objects converted to floats."""
    return JSONResponse(content=_to_json_friendly(content), status_code=status_code)


@app.get("/api/stats")
async def get_stats():
    """
    Aggregated observability metrics.

    Returns:
        total_calls, cache_hits, cache_misses, hit_rate_pct,
        avg_latency_ms, total_cost_usd, error_count, error_rate_pct,
        cache_entry_count, backend_label
    """
    log_store = get_metrics_store()
    cache_store = get_cache_backend()
    stats = log_store.get_stats()
    stats["cache_entry_count"] = cache_store.count()
    stats["backend_label"] = get_active_backend_label()
    return safe_json_response(stats)


@app.get("/api/logs")
async def get_logs(limit: int = Query(default=50, ge=1, le=500)):
    """
    Recent request log entries, newest first.

    Query params:
        limit: Number of rows to return (1–500, default 50)
    """
    log_store = get_metrics_store()
    logs = log_store.get_recent_logs(limit=limit)
    return safe_json_response({"logs": logs, "count": len(logs)})


@app.get("/api/trends")
async def get_trends(
    bucket_minutes: int = Query(default=5, ge=1, le=60),
    buckets: int = Query(default=24, ge=1, le=288),
):
    """
    Time-bucketed metrics for Chart.js charts.

    Query params:
        bucket_minutes: Width of each time bucket (default 5 min)
        buckets: Number of data points to return (default 24 = 2 hours)
    """
    log_store = get_metrics_store()
    trends = log_store.get_trends(bucket_minutes=bucket_minutes, limit_buckets=buckets)
    return safe_json_response({"trends": trends})

# ── Phase 7: Distributed Traces ───────────────────────────────────────────────

@app.get("/api/traces")
async def get_traces(limit: int = Query(default=20, ge=1, le=100)):
    """Recent traces grouped by trace_id with full span lists."""
    log_store = get_metrics_store()
    traces = log_store.get_traces(limit=limit)
    return safe_json_response({"traces": traces, "count": len(traces)})


# ── Phase 9: Canary A-B Stats ─────────────────────────────────────────────────

@app.get("/api/canary")
async def get_canary():
    """Per-variant aggregated metrics for primary vs canary A-B comparison."""
    log_store = get_metrics_store()
    stats = log_store.get_canary_stats()
    return safe_json_response({"variants": stats})
