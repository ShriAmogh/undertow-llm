# undertow-llm

**Wrap any LLM call, get caching, retries, rate limiting, and a real-time dashboard, with zero code changes to your model.**

[![PyPI version](https://img.shields.io/pypi/v/undertow-llm.svg)](https://pypi.org/project/undertow-llm/)
[![Python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://pypi.org/project/undertow-llm/)
[![GitHub Repository](https://img.shields.io/badge/GitHub-ShriAmogh%2Fundertow--llm-blue?logo=github)](https://github.com/ShriAmogh/undertow-llm)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/ShriAmogh/undertow-llm/blob/main/LICENSE)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://pypi.org/project/undertow-llm/)

**Repository**: [https://github.com/ShriAmogh/undertow-llm](https://github.com/ShriAmogh/undertow-llm)

`llm-observability` · `semantic-caching` · `rate-limiting` · `exponential-backoff` · `fallback-chain` · `cost-tracking` · `distributed-tracing` · `openai` · `gemini` · `anthropic` · `ollama` · `pgvector` · `redis`

---

## Why undertow-llm?

Calling LLM APIs in production usually means handling 429 rate limits, transient 500 server errors, duplicate prompt billing, and zero visibility into latency or costs.

Instead of writing custom retry loops and cache wrappers for every provider SDK, `undertow-llm` wraps any Python LLM function with a single `@track()` decorator. You get semantic prompt caching, automated retries, rate limiting, and a local web dashboard without modifying your underlying model calls.

---

## Installation

```bash
pip install undertow-llm
```

Optional extras for specific providers or production backends:

```bash
pip install "undertow-llm[gemini]"    # Google Gemini support
pip install "undertow-llm[ollama]"    # Ollama local model support
pip install "undertow-llm[postgres]"  # PostgreSQL + pgvector backend
pip install "undertow-llm[redis]"     # Redis rate-limiting backend
pip install "undertow-llm[prod]"      # Production stack (Postgres + Redis)
```

---

## Quickstart

**Before** (bare LLM call — no caching, no fallback, no observability):
```python
def generate_response(prompt: str) -> str:
    return client.models.generate_content("gemini-2.5-flash", prompt).text
```

**After** (wrapped with `@track()` — fully resilient & tracked):
```python
from undertow_llm import track

@track(cache=True, retries=3, rate_limit_rate=2.0)
def generate_response(prompt: str) -> str:
    return client.models.generate_content("gemini-2.5-flash", prompt).text
```

No refactoring needed, your function returns its original response while `undertow-llm` handles caching, retries, and background logging automatically.

---

## Real-Time Dashboard

Start the live observability dashboard in one command:

```bash
undertow-llm serve
```

Optionally specify a custom port using `--port` / `-p` (default: `8080`):

```bash
undertow-llm serve -p 9090
```

Open `http://localhost:8080` (or your custom port) to inspect real-time metrics, cache hit ratios, latency charts, cost estimates, distributed traces, and request logs.

---

## How It Works

When a function wrapped with `@track()` is invoked:

1. **Semantic Cache Check**: Computes prompt vector embeddings (`SentenceTransformers`). If a semantically similar prompt exists in storage (above `similarity_threshold`), it returns the cached response in ~15ms with zero API cost.
2. **Rate Limiting**: Enforces token-bucket rates and max concurrency to prevent 429 quota breaches.
3. **Retries & Fallbacks**: If the primary API call raises a transient error, it retries using exponential backoff with jitter. If all retries fail, it executes any configured fallback functions.
4. **Metrics & Logging**: Records call duration, token counts, cost estimates, and trace spans to SQLite (or Postgres + Redis in production).

For a deep dive into the 8-stage execution pipeline, see [ARCHITECTURE.md](https://github.com/ShriAmogh/undertow-llm/blob/main/ARCHITECTURE.md).

---

## Configuration

**Local dev needs zero config** — defaults out-of-the-box to local SQLite (`undertow-llm.db`).

For production environments, configure via environment variables or `configure()`:

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `UNDERTOW_LLM_POSTGRES_URL` / `POSTGRES_URL` | `None` (SQLite) | PostgreSQL URL with `pgvector` for production vector storage & metrics |
| `UNDERTOW_LLM_REDIS_URL` / `REDIS_URL` | `None` (Local) | Redis URL for distributed rate limiting & token buckets |
| `UNDERTOW_LLM_DB_PATH` | `"undertow-llm.db"` | File path for local SQLite database fallback |
| `UNDERTOW_LLM_DASHBOARD_PORT` | `8080` | HTTP port for `undertow-llm serve` dashboard |

### Environment Setup (`.env`)

Copy [.env.example](https://github.com/ShriAmogh/undertow-llm/blob/main/.env.example) to your project `.env` file to enable PostgreSQL and Redis backends:

```env
# Production Storage (Postgres + pgvector for cache & metrics, Redis for rate limits)
POSTGRES_URL=postgresql://postgres:postgres@localhost:5432/undertow_db
REDIS_URL=redis://localhost:6379

# Dashboard Port(Optional)
UNDERTOW_LLM_DASHBOARD_PORT=8080
```

---

## `@track()` Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache` | `bool` | `True` | Enable semantic caching for responses |
| `similarity_threshold` | `float` | `0.92` | Cosine similarity threshold for cache hits (0.0 to 1.0) |
| `cache_ttl` | `int \| None` | `None` | Optional time-to-live in seconds for cached entries |
| `retries` | `int` | `3` | Max retry attempts for transient LLM failures |
| `base_delay` | `float` | `1.0` | Initial exponential backoff delay (seconds) |
| `max_delay` | `float` | `60.0` | Cap on exponential backoff delay (seconds) |
| `jitter` | `bool` | `True` | Add randomized jitter to retry delays to prevent thundering herds |
| `retry_on` | `tuple` | `(Exception,)` | Exception types that trigger automatic retries |
| `fallback` | `list` | `[]` | Ordered list of fallback functions to call if primary function fails |
| `rate_limit_rate` | `float` | `2.0` | Token-bucket refill rate (tokens/second) |
| `rate_limit_max_tokens` | `float` | `10.0` | Token-bucket maximum capacity (burst limit) |
| `max_concurrency` | `int \| None` | `None` | Max concurrent executions allowed across processes |
| `cost_per_call` | `float \| None` | `None` | Explicit cost override per call ($/call) |
| `policy` | `callable \| None` | `None` | Custom safety hook returning `"allow"`, `"block"`, or `"flag"` |
| `canary` | `dict \| None` | `None` | Canary routing config `{"fn": alternate_fn, "weight": 0.10}` |

---

## Supported Providers

`undertow_llm` works with **any provider** -  OpenAI, Anthropic, Google Gemini, Ollama, HuggingFace, or custom local models — since it wraps your existing Python function call rather than a specific provider SDK.

---

## Examples

See [`demo/example_usage.py`](https://github.com/ShriAmogh/undertow-llm/blob/main/demo/example_usage.py) for complete runnable examples.

### 1. Multi-Provider Fallback Chain
```python
from undertow_llm import track

def fallback_anthropic(prompt: str) -> str:
    return anthropic_client.messages.create(model="claude-3-5-sonnet", messages=[{"role": "user", "content": prompt}]).content[0].text

@track(retries=2, fallback=[fallback_anthropic])
def primary_openai(prompt: str) -> str:
    return openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}]).choices[0].message.content
```

### 2. Streaming LLM Response
```python
@track(cache=False)
def stream_gemini(prompt: str):
    response = gemini_client.models.generate_content_stream("gemini-2.5-flash", prompt)
    for chunk in response:
        yield chunk.text
```

### 3. Local Model (Ollama) with Custom Usage Extractor
```python
@track(
    cost_per_call=0.0,  # Local model — zero API cost
    usage_extractor=lambda res: {"prompt_tokens": len(res.get("prompt", "")), "completion_tokens": len(res.get("response", ""))}
)
def ask_ollama(prompt: str) -> dict:
    return ollama.generate(model="llama3", prompt=prompt)
```

---

## Known Limitations & Roadmap

### Limitations
- **SQLite Concurrency**: Local SQLite storage (`undertow-llm.db`) is zero-config for development, but high concurrency multi-node deployments require setting `POSTGRES_URL` and `REDIS_URL`.


---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/ShriAmogh/undertow-llm/blob/main/CONTRIBUTING.md) for developer setup details.

---

## License

MIT — see [LICENSE](https://github.com/ShriAmogh/undertow-llm/blob/main/LICENSE)
