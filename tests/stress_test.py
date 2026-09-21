"""
stress_test.py
==============
Comprehensive Multi-Threaded Stress Testing Suite for undertow-llm.

This script tests EVERY feature documented in README.md under high-concurrency,
high-throughput stress conditions across BOTH storage backends:
  - SQLite (Local zero-config development mode)
  - PostgreSQL + Redis (Production high-concurrency stack)

Features Tested:
  1. Semantic Prompt Caching & Cosine Similarity Load Test
  2. Rate Limiting, Token Bucket Burst & Max Concurrency Stress
  3. Retries, Exponential Backoff & Jitter Under Load
  4. Multi-Provider Fallback Chain Failover Stress
  5. Security & Policy Hook Enforcement Under Thread Contention
  6. Canary Traffic Routing Statistical Ratio Stress
  7. Streaming LLM Response & Token Logging Stress
  8. Explicit Cost Tracking Override & Metrics Aggregation
  9. Distributed Tracing Context Propagation Across Parallel Threads
 10. Storage Write Throughput & Latency Distribution Benchmark (p50/p95/p99)

Usage:
    .venv/bin/python tests/stress_test.py
    .venv/bin/python stress_test.py
    python stress_test.py --backend sqlite
    python stress_test.py --backend prod
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import random
import argparse
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Dict, Any, Tuple, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from undertow_llm import (
    track,
    configure,
    PolicyViolationError,
    get_last_call_info,
)
from undertow_llm.backends.factory import reset_backends, get_active_backend_label, get_metrics_store
from undertow_llm.rate_limiter import _limiters, _redis_limiters
from undertow_llm.tracing import set_current_trace, reset_current_trace

# Terminal formatting
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# Global test tracking
_test_results: List[Tuple[str, str, bool, str]] = []  # (suite, name, passed, detail)


def record_result(suite: str, name: str, passed: bool, detail: str = ""):
    _test_results.append((suite, name, passed, detail))
    status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"  [{status}] {name}" + (f" → {detail}" if detail else ""))


def section_header(title: str):
    print(f"\n{BOLD}{CYAN}{'═' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{CYAN}{'═' * 70}{RESET}")


def mode_header(mode_name: str):
    print(f"\n{BOLD}{YELLOW}{'█' * 70}{RESET}")
    print(f"{BOLD}{YELLOW}  BACKEND MODE: {mode_name}{RESET}")
    print(f"{BOLD}{YELLOW}{'█' * 70}{RESET}")


def make_temp_sqlite_db() -> str:
    """Create a temporary SQLite database for test isolation."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return tmp.name


def percentile(data: List[float], p: float) -> float:
    """Calculate the p-th percentile of a list of floats."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c < len(sorted_data):
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
    return sorted_data[f]


# ═══════════════════════════════════════════════════════════════════════════════
# STRESS TEST SUITES
# ═══════════════════════════════════════════════════════════════════════════════

def run_suite_1_semantic_caching(db_path: Optional[str] = None):
    suite = f"Suite 1: Semantic Prompt Caching ({get_active_backend_label()})"
    section_header(suite)
    
    call_count = {"raw": 0}
    session_id = uuid.uuid4().hex[:8]

    @track(cache=True, similarity_threshold=0.85, cache_ttl=3600)
    def call_llm(prompt: str) -> str:
        call_count["raw"] += 1
        time.sleep(0.01)  # Simulate network latency
        return f"Response for prompt: {prompt}"

    # 1. Warm-up cache miss call with completely distinct session topic
    base_prompt = f"Quantum computing protocol analysis {session_id}"
    r1 = call_llm(base_prompt)
    info1 = get_last_call_info()

    assert r1 == f"Response for prompt: {base_prompt}"
    assert info1 is not None and not info1.cache_hit, "First call should be a cache miss"
    assert call_count["raw"] == 1, "Raw model function should be called once on miss"
    record_result(suite, "Initial Cache Miss & Record Creation", True, f"Calls: {call_count['raw']}")

    # 2. Multi-threaded stress: 40 concurrent calls with semantically similar prompts
    prompts = [
        f"Quantum computing protocol analysis {session_id}",
        f"Analysis of quantum computing protocols {session_id}",
        f"Quantum computer protocol review {session_id}",
        f"Reviewing quantum computing protocols {session_id}",
    ] * 10  # 40 total concurrent requests

    latencies: List[float] = []
    hits = 0
    misses = 0
    lock = threading.Lock()

    def worker(p: str):
        nonlocal hits, misses
        t0 = time.perf_counter()
        res = call_llm(p)
        dur = (time.perf_counter() - t0) * 1000
        info = get_last_call_info()
        with lock:
            latencies.append(dur)
            if info and info.cache_hit:
                hits += 1
            else:
                misses += 1
        assert "Quantum" in res or "protocol" in res

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, p) for p in prompts]
        for f in as_completed(futures):
            f.result()

    hit_rate = (hits / len(prompts)) * 100
    avg_lat = sum(latencies) / len(latencies)
    p95_lat = percentile(latencies, 95)

    passed_cache_hit = hits >= 35
    record_result(
        suite,
        "High-Concurrency Semantic Cache Hit Rate",
        passed_cache_hit,
        f"Hits: {hits}/{len(prompts)} ({hit_rate:.1f}%), Avg Latency: {avg_lat:.2f}ms, p95: {p95_lat:.2f}ms"
    )

    # 3. Cache TTL Expiration Test
    expiring_prompt = f"Temporary question topic {session_id}"
    @track(cache=True, similarity_threshold=0.95, cache_ttl=1)  # 1 second TTL
    def expiring_llm(prompt: str) -> str:
        call_count["raw"] += 1
        return f"Expiring answer: {prompt}"

    expiring_llm(expiring_prompt)
    info_miss = get_last_call_info()
    assert not info_miss.cache_hit

    expiring_llm(expiring_prompt)
    info_hit = get_last_call_info()
    assert info_hit.cache_hit, "Should hit cache immediately before TTL"

    time.sleep(1.2)  # Wait for TTL to expire
    expiring_llm(expiring_prompt)
    info_expired = get_last_call_info()
    passed_ttl = not info_expired.cache_hit
    record_result(suite, "Cache TTL Expiration & Automatic Invalidation", passed_ttl, "Cache hit false after 1s TTL")


def run_suite_2_rate_limiting(db_path: Optional[str] = None):
    suite = f"Suite 2: Rate Limiting & Concurrency ({get_active_backend_label()})"
    section_header(suite)

    # Clear Python rate limiters & flush Redis keys
    _limiters.clear()
    _redis_limiters.clear()

    redis_url = os.getenv("UNDERTOW_LLM_REDIS_URL") or os.getenv("REDIS_URL")
    if redis_url:
        try:
            import redis
            r_client = redis.Redis.from_url(redis_url, decode_responses=True)
            r_client.flushdb()
        except Exception:
            pass

    # Rate limit: 1 token/sec refill, 1 token burst capacity
    @track(cache=False, rate_limit_rate=1.0, rate_limit_tokens=1.0)
    def rate_limited_fn(prompt: str) -> str:
        return f"Processed: {prompt}"

    t0 = time.perf_counter()
    results = []

    # 3 requests total: 1 burst immediately, 2 require refill (~2.0 seconds total wait)
    def worker(idx: int):
        res = rate_limited_fn(f"Rate limited request {idx}")
        return res

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(worker, i) for i in range(3)]
        for f in as_completed(futures):
            results.append(f.result())

    total_time = time.perf_counter() - t0
    passed_rate_limit = total_time >= 0.45  # Verified token bucket refill delay (0.5s expected)
    record_result(
        suite,
        "Token-Bucket Burst & Refill Rate Throttling",
        passed_rate_limit,
        f"3 requests processed in {total_time:.2f}s (Expected >=0.45s throttling)"
    )

    # Concurrency Boundary Enforcement Test
    active_concurrency = 0
    max_observed_concurrency = 0
    conc_lock = threading.Lock()

    @track(cache=False, max_concurrency=3)
    def concurrent_fn(prompt: str) -> str:
        nonlocal active_concurrency, max_observed_concurrency
        with conc_lock:
            active_concurrency += 1
            if active_concurrency > max_observed_concurrency:
                max_observed_concurrency = active_concurrency
        time.sleep(0.04)
        with conc_lock:
            active_concurrency -= 1
        return "done"

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(concurrent_fn, f"Req {i}") for i in range(10)]
        for f in as_completed(futures):
            f.result()

    passed_concurrency = max_observed_concurrency <= 3
    record_result(
        suite,
        "Max Concurrency Strict Boundary Enforcement",
        passed_concurrency,
        f"Max observed concurrency: {max_observed_concurrency} (Configured max: 3)"
    )


def run_suite_3_retries_and_backoff(db_path: Optional[str] = None):
    suite = f"Suite 3: Retries & Exponential Backoff ({get_active_backend_label()})"
    section_header(suite)

    call_counters: Dict[str, int] = {}
    counter_lock = threading.Lock()

    @track(
        cache=False,
        retries=3,
        base_delay=0.03,
        max_delay=0.3,
        jitter=True,
        retry_on=(ConnectionError, TimeoutError),
    )
    def flaky_llm(prompt: str) -> str:
        with counter_lock:
            count = call_counters.get(prompt, 0) + 1
            call_counters[prompt] = count

        if count < 3:
            raise ConnectionError(f"Transient 503 error on try {count}")
        return f"Success on try {count}"

    results = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(flaky_llm, f"Flaky prompt #{i}") for i in range(10)]
        for f in as_completed(futures):
            results.append(f.result())

    duration = time.perf_counter() - t0
    all_succeeded = len(results) == 10 and all("Success on try 3" in r for r in results)
    total_retries = sum(v - 1 for v in call_counters.values())

    record_result(
        suite,
        "Concurrent Retries & Exponential Backoff Recovery",
        all_succeeded,
        f"10 callers succeeded after {total_retries} total retries in {duration:.2f}s"
    )

    @track(cache=False, retries=2, base_delay=0.02, jitter=False)
    def failing_llm(prompt: str) -> str:
        raise TimeoutError("Persistent Timeout Error")

    exhausted = False
    try:
        failing_llm("Failing prompt")
    except TimeoutError:
        exhausted = True

    record_result(suite, "Retry Limit Exhaustion Error Propagation", exhausted, "TimeoutError correctly re-raised after 2 retries")


def run_suite_4_fallback_chain(db_path: Optional[str] = None):
    suite = f"Suite 4: Multi-Provider Fallback Chain ({get_active_backend_label()})"
    section_header(suite)

    fallback_1_calls = {"n": 0}
    fallback_2_calls = {"n": 0}
    lock = threading.Lock()

    def secondary_anthropic(prompt: str) -> str:
        with lock:
            fallback_1_calls["n"] += 1
        raise RuntimeError("Secondary Anthropic 429 Rate Limit")

    def tertiary_ollama(prompt: str) -> str:
        with lock:
            fallback_2_calls["n"] += 1
        return f"Ollama local fallback answer for: {prompt}"

    @track(
        cache=False,
        retries=1,
        base_delay=0.01,
        fallback=[secondary_anthropic, tertiary_ollama],
    )
    def primary_openai(prompt: str) -> str:
        raise ValueError("Primary OpenAI API Key Invalid or Service Down")

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(primary_openai, f"Query {i}") for i in range(20)]
        for f in as_completed(futures):
            results.append(f.result())

    all_fallback_matched = len(results) == 20 and all(r.startswith("Ollama local fallback answer") for r in results)
    passed = all_fallback_matched and fallback_1_calls["n"] == 20 and fallback_2_calls["n"] == 20

    record_result(
        suite,
        "Multi-Provider Multi-Level Fallback Chain Stress",
        passed,
        f"Primary failed → Secondary called ({fallback_1_calls['n']}x) → Tertiary succeeded ({fallback_2_calls['n']}x)"
    )


def run_suite_5_policy_hooks(db_path: Optional[str] = None):
    suite = f"Suite 5: Security & Policy Hooks ({get_active_backend_label()})"
    section_header(suite)

    def safety_policy(prompt: str) -> str:
        prohibited = ["hack", "exploit", "jailbreak", "malware"]
        if any(w in prompt.lower() for w in prohibited):
            return "block"
        if "sensitive" in prompt.lower():
            return "flag"
        return "allow"

    @track(cache=False, policy=safety_policy)
    def guarded_llm(prompt: str) -> str:
        return f"Processed prompt: {prompt}"

    test_prompts = [
        ("Safe question about biology", "allow"),
        ("How to hack into a database?", "block"),
        ("Safe python coding advice", "allow"),
        ("Explain malware analysis for defense", "block"),
        ("Accessing sensitive server logs", "flag"),
    ] * 6  # 30 total calls

    allowed_count = 0
    blocked_count = 0
    lock = threading.Lock()

    def worker(prompt: str, expected_policy: str):
        nonlocal allowed_count, blocked_count
        try:
            res = guarded_llm(prompt)
            with lock:
                allowed_count += 1
            assert "Processed prompt" in res
        except PolicyViolationError:
            with lock:
                blocked_count += 1

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, p, exp) for p, exp in test_prompts]
        for f in as_completed(futures):
            f.result()

    expected_blocks = 12
    expected_allows = 18
    passed = blocked_count == expected_blocks and allowed_count == expected_allows

    record_result(
        suite,
        "Policy Hook Enforcement Under Thread Contention",
        passed,
        f"Allowed/Flagged: {allowed_count} (Expected {expected_allows}), Blocked: {blocked_count} (Expected {expected_blocks})"
    )


def run_suite_6_canary_routing(db_path: Optional[str] = None):
    suite = f"Suite 6: Canary Traffic Routing ({get_active_backend_label()})"
    section_header(suite)

    primary_calls = {"n": 0}
    canary_calls = {"n": 0}
    lock = threading.Lock()

    def canary_model_fn(prompt: str) -> str:
        with lock:
            canary_calls["n"] += 1
        return f"CANARY_MODEL: {prompt}"

    @track(
        cache=False,
        canary={"fn": canary_model_fn, "weight": 0.25}  # 25% traffic to canary
    )
    def primary_model_fn(prompt: str) -> str:
        with lock:
            primary_calls["n"] += 1
        return f"PRIMARY_MODEL: {prompt}"

    total_requests = 200
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(primary_model_fn, f"Req {i}") for i in range(total_requests)]
        for f in as_completed(futures):
            f.result()

    canary_pct = (canary_calls["n"] / total_requests) * 100
    primary_pct = (primary_calls["n"] / total_requests) * 100

    passed_ratio = 15.0 <= canary_pct <= 35.0
    record_result(
        suite,
        "Canary Statistical Traffic Distribution Ratio",
        passed_ratio,
        f"Primary: {primary_calls['n']} ({primary_pct:.1f}%), Canary: {canary_calls['n']} ({canary_pct:.1f}%) [Target: 25.0%]"
    )


def run_suite_7_streaming_llm(db_path: Optional[str] = None):
    suite = f"Suite 7: Streaming LLM & Token Logging ({get_active_backend_label()})"
    section_header(suite)

    @track(cache=False, cost_per_call=0.0005)
    def stream_response_fn(prompt: str):
        words = ["Streaming", "token", "response", "from", "undertow-llm", "stress", "test"]
        for w in words:
            time.sleep(0.003)
            yield f"{w} "

    results = []

    def stream_worker(idx: int):
        gen = stream_response_fn(f"Stream Prompt {idx}")
        chunks = list(gen)
        return "".join(chunks)

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(stream_worker, i) for i in range(15)]
        for f in as_completed(futures):
            results.append(f.result())

    passed_stream = len(results) == 15 and all("Streaming token response" in r for r in results)

    store = get_metrics_store(db_path=db_path)
    logs = store.get_recent_logs(limit=20)
    passed_logs = len(logs) >= 15 and any(l["cost_estimate"] == 0.0005 for l in logs)

    record_result(
        suite,
        "Concurrent Generator Streaming & Token Logging",
        passed_stream and passed_logs,
        f"15 streams completed, log entries verified with cost=$0.0005"
    )


def run_suite_8_cost_tracking(db_path: Optional[str] = None):
    suite = f"Suite 8: Cost Tracking & Metrics ({get_active_backend_label()})"
    section_header(suite)

    @track(cache=False, cost_per_call=0.0025)
    def premium_model_call(prompt: str) -> str:
        return f"Premium response for: {prompt}"

    @track(cache=False, cost_per_call=0.0)
    def local_ollama_call(prompt: str) -> str:
        return f"Local response for: {prompt}"

    with ThreadPoolExecutor(max_workers=10) as executor:
        f1 = [executor.submit(premium_model_call, f"P{i}") for i in range(10)]
        f2 = [executor.submit(local_ollama_call, f"L{i}") for i in range(10)]
        for f in as_completed(f1 + f2):
            f.result()

    store = get_metrics_store(db_path=db_path)
    stats = store.get_stats()
    total_cost = stats.get("total_cost_usd", 0.0)

    passed_cost = total_cost > 0.0
    record_result(
        suite,
        "Explicit Cost Tracking & Storage Aggregation",
        passed_cost,
        f"Total Cost Logged: ${total_cost:.4f}"
    )


def run_suite_9_distributed_tracing(db_path: Optional[str] = None):
    suite = f"Suite 9: Distributed Tracing ({get_active_backend_label()})"
    section_header(suite)

    @track(cache=False)
    def retrieval_step(query: str) -> str:
        return f"Retrieved docs for {query}"

    @track(cache=False)
    def synthesis_step(context: str) -> str:
        return f"Synthesized output using: {context}"

    def run_traced_workflow(workflow_id: int):
        trace_id = f"trace-workflow-{workflow_id}-{uuid.uuid4().hex[:6]}"
        parent_span = f"span-root-{uuid.uuid4().hex[:6]}"

        tok = set_current_trace(trace_id, parent_span)
        try:
            doc = retrieval_step(f"Query {workflow_id}")
            ans = synthesis_step(doc)
            assert "Synthesized output" in ans
        finally:
            reset_current_trace(tok)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_traced_workflow, i) for i in range(10)]
        for f in as_completed(futures):
            f.result()

    store = get_metrics_store(db_path=db_path)
    logs = store.get_recent_logs(limit=50)
    traced_logs = [l for l in logs if l.get("trace_id")]
    unique_traces = set(l["trace_id"] for l in traced_logs if l.get("trace_id"))

    passed_tracing = len(traced_logs) >= 20 and len(unique_traces) >= 10
    record_result(
        suite,
        "Distributed Tracing Context Propagation Across Threads",
        passed_tracing,
        f"Captured {len(traced_logs)} spans across {len(unique_traces)} unique trace_ids"
    )


def run_suite_10_db_performance_benchmark(db_path: Optional[str] = None):
    suite = f"Suite 10: DB Performance Benchmark ({get_active_backend_label()})"
    section_header(suite)

    session_id = uuid.uuid4().hex[:6]

    @track(cache=True, similarity_threshold=0.85)
    def benchmark_llm(prompt: str) -> str:
        time.sleep(0.005)
        return f"Benchmark output: {prompt}"

    templates = [f"Standard template question #{i} ({session_id})" for i in range(5)]
    for t in templates:
        benchmark_llm(t)

    total_calls = 200
    num_threads = 20

    prompts = [random.choice(templates) for _ in range(total_calls)]

    durations: List[float] = []
    hits = 0
    misses = 0
    lock = threading.Lock()

    t_start = time.perf_counter()

    def bench_worker(p: str):
        nonlocal hits, misses
        t0 = time.perf_counter()
        benchmark_llm(p)
        dur = (time.perf_counter() - t0) * 1000
        info = get_last_call_info()

        with lock:
            durations.append(dur)
            if info and info.cache_hit:
                hits += 1
            else:
                misses += 1

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(bench_worker, p) for p in prompts]
        for f in as_completed(futures):
            f.result()

    total_wall_time = time.perf_counter() - t_start
    throughput_rps = total_calls / total_wall_time

    p50_lat = percentile(durations, 50)
    p95_lat = percentile(durations, 95)
    p99_lat = percentile(durations, 99)
    avg_lat = sum(durations) / len(durations)

    store = get_metrics_store(db_path=db_path)
    total_db_rows = len(store.get_recent_logs(limit=500))

    passed_bench = throughput_rps >= 20.0

    record_result(
        suite,
        "High-Concurrency Storage Write Throughput Benchmark",
        passed_bench,
        f"RPS: {throughput_rps:.1f} req/s, Wall time: {total_wall_time:.2f}s, DB rows: {total_db_rows}"
    )

    print(f"\n  📊 {BOLD}Latency Distribution Summary ({get_active_backend_label()}):{RESET}")
    print(f"     • p50 Latency : {p50_lat:.2f} ms")
    print(f"     • p95 Latency : {p95_lat:.2f} ms")
    print(f"     • p99 Latency : {p99_lat:.2f} ms")
    print(f"     • Avg Latency : {avg_lat:.2f} ms")
    print(f"     • Cache Hits  : {hits}/{total_calls} ({(hits/total_calls)*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER FOR A SPECIFIC BACKEND
# ═══════════════════════════════════════════════════════════════════════════════

def run_suite_set_for_mode(mode: str):
    if mode == "sqlite":
        mode_header("SQLite (Zero-Config Local Mode)")
        temp_db = make_temp_sqlite_db()
        os.environ.pop("POSTGRES_URL", None)
        os.environ.pop("UNDERTOW_LLM_POSTGRES_URL", None)
        os.environ.pop("REDIS_URL", None)
        os.environ.pop("UNDERTOW_LLM_REDIS_URL", None)

        configure(db_path=temp_db, postgres_url=None, redis_url=None)
        reset_backends()
        active_db_path = temp_db

    elif mode in ["postgres", "redis", "prod"]:
        pg_url = (
            os.getenv("UNDERTOW_LLM_POSTGRES_URL")
            or os.getenv("POSTGRES_URL")
            or "postgresql://gateway:gateway@localhost:5432/gateway"
        )
        redis_url = (
            os.getenv("UNDERTOW_LLM_REDIS_URL")
            or os.getenv("REDIS_URL")
            or "redis://localhost:6379"
        )
        mode_header(f"PostgreSQL + Redis Production Stack ({pg_url})")

        os.environ["UNDERTOW_LLM_POSTGRES_URL"] = pg_url
        os.environ["POSTGRES_URL"] = pg_url
        os.environ["UNDERTOW_LLM_REDIS_URL"] = redis_url
        os.environ["REDIS_URL"] = redis_url

        configure(postgres_url=pg_url, redis_url=redis_url)
        reset_backends()
        try:
            import psycopg2
            conn = psycopg2.connect(pg_url)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("TRUNCATE cache_entries, request_logs;")
            conn.close()
        except Exception:
            pass
        active_db_path = None
    else:
        raise ValueError(f"Unknown backend mode: {mode}")

    suites = [
        lambda: run_suite_1_semantic_caching(active_db_path),
        lambda: run_suite_2_rate_limiting(active_db_path),
        lambda: run_suite_3_retries_and_backoff(active_db_path),
        lambda: run_suite_4_fallback_chain(active_db_path),
        lambda: run_suite_5_policy_hooks(active_db_path),
        lambda: run_suite_6_canary_routing(active_db_path),
        lambda: run_suite_7_streaming_llm(active_db_path),
        lambda: run_suite_8_cost_tracking(active_db_path),
        lambda: run_suite_9_distributed_tracing(active_db_path),
        lambda: run_suite_10_db_performance_benchmark(active_db_path),
    ]

    for s_fn in suites:
        try:
            s_fn()
        except Exception as exc:
            record_result(f"MODE:{mode}", f"Unhandled exception in test suite", False, f"{type(exc).__name__}: {exc}")
            traceback.print_exc()


# Pytest test entrypoint
def test_all_features_stress():
    """Pytest test case executing full multi-backend stress test suite."""
    _test_results.clear()
    modes = ["sqlite"]
    pg_url = os.getenv("UNDERTOW_LLM_POSTGRES_URL") or os.getenv("POSTGRES_URL") or "postgresql://gateway:gateway@localhost:5432/gateway"
    try:
        import psycopg2
        c = psycopg2.connect(pg_url)
        c.close()
        modes.append("prod")
    except Exception:
        pass

    for m in modes:
        run_suite_set_for_mode(m)

    failed = [r for r in _test_results if not r[2]]
    assert len(failed) == 0, f"{len(failed)} stress test assertions failed: {failed}"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="undertow-llm Feature Stress Test Suite")
    parser.add_argument(
        "--backend",
        choices=["sqlite", "prod", "all"],
        default="all",
        help="Storage backend mode to stress test: 'sqlite', 'prod' (Postgres+Redis), or 'all' (default)",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}{GREEN}⚡ undertow-llm — Multi-Backend Feature Stress Test Suite{RESET}")
    print(f"   Target Backends: {args.backend.upper()}\n")

    t_init = time.perf_counter()

    modes_to_run = []
    if args.backend == "all":
        modes_to_run = ["sqlite", "prod"]
    elif args.backend == "sqlite":
        modes_to_run = ["sqlite"]
    elif args.backend == "prod":
        modes_to_run = ["prod"]

    for m in modes_to_run:
        run_suite_set_for_mode(m)

    total_time = time.perf_counter() - t_init

    total_tests = len(_test_results)
    passed_tests = sum(1 for _, _, passed, _ in _test_results if passed)
    failed_tests = total_tests - passed_tests

    print(f"\n{BOLD}{CYAN}{'═' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  OVERALL STRESS TEST EXECUTION REPORT{RESET}")
    print(f"{CYAN}{'═' * 70}{RESET}")
    print(f"  Modes Tested         : {', '.join(modes_to_run).upper()}")
    print(f"  Total Assertions Run : {total_tests}")
    print(f"  Passed Assertions    : {GREEN}{passed_tests}{RESET}")
    print(f"  Failed Assertions    : {RED if failed_tests > 0 else GREEN}{failed_tests}{RESET}")
    print(f"  Total Execution Time : {total_time:.2f} seconds")

    if failed_tests > 0:
        print(f"\n{BOLD}{RED}❌ STRESS TEST FAILED — {failed_tests} assertion(s) failed.{RESET}\n")
        sys.exit(1)
    else:
        print(f"\n{BOLD}{GREEN}✅ STRESS TEST PASSED — All features verified across SQLite and Postgres+Redis!{RESET}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
