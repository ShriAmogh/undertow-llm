# gateway-sdk Architecture

This document provides a deep dive into the design and internal lifecycle of `gateway-sdk`.

## Overview

`gateway-sdk` is a provider-agnostic LLM observability and reliability layer designed as a Python function decorator (`@track()`). Unlike traditional API proxies or gateway sidecars, `gateway-sdk` operates directly within your application process, avoiding network overhead, complex proxy deployments, and provider-specific locking.

```
Your LLM Function
      │
  @track() Decorator
      │
  ┌───┴───────────────────────────────────────────────────────┐
  │  Lifecycle Orchestration                                  │
  │  1. Safety policy evaluation                              │
  │  2. Semantic cache lookup (cosine similarity)              │
  │  3. Canary A/B traffic split                              │
  │  4. Distributed token-bucket rate limiting               │
  │  5. Execution with exponential backoff & jitter retries   │
  │  6. Fallback provider failover execution                   │
  │  7. Async metric logging & vector persistence             │
  └───────────────────────────────────────────────────────────┘
      │
  ┌───┴───────────────────────────────────────────────────────┐
  │  Pluggable Storage Layer (Factory Dispatcher)             │
  │  ├── SQLite (zero-config local dev default)               │
  │  ├── PostgreSQL + pgvector (production metrics & vectors) │
  │  └── Redis (distributed rate limiting & queues)           │
  └───────────────────────────────────────────────────────────┘
      │
  FastAPI Dashboard (`gateway-sdk serve`)
```

## Core Components

### 1. Decorator Lifecycle (`gateway_sdk.decorator`)
The `@track()` decorator wraps target function calls and executes an 8-stage lifecycle sequence:
- **Policy Check**: Evaluates user-defined safety callbacks before execution.
- **Cache Lookup**: Computes query embeddings (via SentenceTransformers `all-MiniLM-L6-v2`) and searches vector store for cosine similarity $\ge \text{threshold}$.
- **Canary Routing**: Optionally routes a percentage of calls to alternative functions for A/B testing.
- **Rate Limiting**: Enforces token-bucket rate limits (local in-memory or Redis-backed).
- **Execution & Retries**: Executes the underlying function with exponential backoff and jitter.
- **Fallback Execution**: If primary retries fail, attempts registered fallback functions in sequence.
- **Metric Logging**: Extracts token counts, latency, cost estimates, and traces.
- **Async Persistence**: Asynchronously writes log entries and vector embeddings to storage.

### 2. Storage Backend Factory (`gateway_sdk.backends`)
`gateway-sdk` uses an abstract interface (`CacheBackend` and `MetricsStore`) with factory auto-discovery:
- **SQLite (`SQLiteCacheBackend` / `SQLiteMetricsStore`)**: Zero-config storage storing SQLite vectors and log tables in `gateway.db`.
- **PostgreSQL (`PostgresCacheBackend` / `PostgresMetricsStore`)**: Uses `pgvector` extension for fast vector similarity search and relational analytics. Automatically activates when `GATEWAY_SDK_POSTGRES_URL` or `POSTGRES_URL` is configured.
- **Redis Integration**: Uses Redis for distributed rate limit counters and queue locks when `GATEWAY_SDK_REDIS_URL` is set.

### 3. FastAPI Dashboard (`gateway_sdk.server`)
A lightweight FastAPI server (`gateway-sdk serve`) serving a dynamic HTML5 dashboard with real-time Chart.js charts, log viewers, trace waterfalls, and canary analytics.
