# gateway-sdk

**Wrap any LLM call, get caching, retries, rate limiting, and a real-time dashboard — with zero code changes to your model.**

[![PyPI version](https://img.shields.io/pypi/v/gateway-sdk)](https://pypi.org/project/gateway-sdk/)
[![Python](https://img.shields.io/pypi/pyversions/gateway-sdk)](https://pypi.org/project/gateway-sdk/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/amogharora/gateway-sdk/ci.yml?branch=main)](https://github.com/amogharora/gateway-sdk/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Problem Statement

Without an observability and resilience layer, LLM applications suffer from soaring API costs due to redundant prompt calls, unexpected provider outages with zero fallback resilience, and complete lack of visibility into latency and errors. `gateway-sdk` solves this by wrapping your existing Python LLM functions in a single decorator—providing semantic caching, automated retries, rate limiting, and a live dashboard without modifying your model logic.

---

## Install

```bash
pip install gateway-sdk
```

---

## Quickstart

**Before** (bare LLM call — no caching, no fallback, no observability):
```python
def generate_text(prompt: str) -> str:
    return client.models.generate_content("gemini-2.5-flash", prompt).text
```

**After** (wrapped with `@track()` — fully resilient & tracked):
```python
from gateway_sdk import track

@track(cache=True, retries=3, rate_limit_rate=2.0)
def generate_text(prompt: str) -> str:
    return client.models.generate_content("gemini-2.5-flash", prompt).text
```

Copy-paste into your application and run. Zero edits required except setting your provider API key.

---

## Dashboard

Start the live observability dashboard in one command:

```bash
gateway-sdk serve
```

Open `http://localhost:8080` to inspect real-time metrics, cache hit ratios, latency charts, cost estimates, and request logs.

![gateway-sdk Dashboard Preview](https://raw.githubusercontent.com/amogharora/gateway-sdk/main/docs/dashboard_preview.png)

---

## How It Works

1. The `@track()` decorator wraps your function, intercepting incoming prompts before execution.
2. It performs a vector similarity search to serve semantic cache hits instantly and applies token-bucket rate limits.
3. Upon function completion, it records latency, token usage, estimated cost, and execution traces to storage.
4. It is provider-agnostic because it wraps your Python function call directly and never touches your underlying model SDK.

For a detailed architectural breakdown of the 8-stage execution pipeline and backend dispatcher, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Configuration

**Local dev needs zero config** — defaults out-of-the-box to local SQLite (`gateway.db`).

For production environments, configure via environment variables:

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `GATEWAY_SDK_POSTGRES_URL` | `None` (SQLite) | PostgreSQL URL with `pgvector` for production vector storage & metrics |
| `GATEWAY_SDK_REDIS_URL` | `None` (Local) | Redis URL for distributed rate limiting & token buckets |
| `GATEWAY_DB_PATH` | `"gateway.db"` | File path for local SQLite database fallback |
| `GATEWAY_DASHBOARD_PORT` | `8080` | HTTP port for `gateway-sdk serve` dashboard |

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

`gateway-sdk` works with **any provider** — OpenAI, Anthropic, Google Gemini, Ollama, HuggingFace, or custom local models — since it wraps your existing Python function call rather than a specific provider SDK.

---

## Examples

See [`demo/example_usage.py`](demo/example_usage.py) for complete runnable examples.

### 1. Multi-Provider Fallback Chain
```python
from gateway_sdk import track

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
- **SQLite Concurrency**: Local SQLite storage (`gateway.db`) is zero-config and ideal for development and single-instance apps, but is not designed for multi-node production scale. For high concurrency, set `GATEWAY_SDK_POSTGRES_URL` and `GATEWAY_SDK_REDIS_URL`.

### Near-Term Roadmap
- [ ] OpenTelemetry trace exporter integration
- [ ] Multi-tenant workspace tagging & dashboard authentication
- [ ] Automated PII redaction and sensitive prompt masking filters

---

## Contributing

Contributions are welcome! Please run unit tests before submitting pull requests:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full developer details.

---

## License

MIT — see [LICENSE](LICENSE)
