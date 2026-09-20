"""
complete_testing.py
===================
End-to-end integration test script for lensllm (Phases 1–4).

This script tests REAL components (no mocks) against a temporary SQLite DB,
verifying the full call flow from @track() → embed → cache → retry → log.

Usage:
    cd /Users/amogharora/task_decorator
    source .venv/bin/activate
    python complete_testing.py

Output: colour-coded PASS/FAIL summary for every test group.
Exit code: 0 if all pass, 1 if any fail.
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
import threading
import traceback

# Ensure SQLite test isolation for standalone script execution
os.environ.pop("POSTGRES_URL", None)
os.environ.pop("LENSLLM_POSTGRES_URL", None)
from lensllm import configure
from lensllm.backends.factory import reset_backends
configure(postgres_url=None)
reset_backends()
from typing import Callable

# pytest.approx is used in Phase 4 timing assertions
try:
    import pytest
except ImportError:
    # Minimal pytest.approx stand-in so the script works without pytest installed
    class _Approx:
        def __init__(self, v, abs=0.1, rel=None): self.v = v; self._abs = abs
        def __eq__(self, other): return abs(other - self.v) <= self._abs
        def __repr__(self): return f"approx({self.v}, abs={self._abs})"
    class _Pytest:
        def approx(self, v, abs=0.1, rel=None): return _Approx(v, abs=abs)
    pytest = _Pytest()


# ── Ensure project root is on path ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("LENSLLM_DB_PATH", ":memory:")  # overridden per test

# ── Terminal colours ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ── Result tracking ───────────────────────────────────────────────────────────
_results: list[tuple[str, str, bool, str]] = []   # (group, name, passed, detail)


def record(group: str, name: str, passed: bool, detail: str = ""):
    _results.append((group, name, passed, detail))
    status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"  [{status}] {name}" + (f"  → {detail}" if detail else ""))


def section(title: str):
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{CYAN}{'─' * 60}{RESET}")


def run_test(group: str, name: str, fn: Callable):
    """Run a test function and record pass/fail."""
    try:
        fn()
        record(group, name, True)
    except AssertionError as e:
        record(group, name, False, str(e))
    except Exception as e:
        record(group, name, False, f"{type(e).__name__}: {e}")


# ── Shared temp DB factory ────────────────────────────────────────────────────
def make_temp_db() -> str:
    """Create a fresh temporary SQLite DB and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return tmp.name


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Config & DB Schema
# ═══════════════════════════════════════════════════════════════════════════════

section("Phase 1 — Config & Database Schema")

from lensllm.config import configure, get_config, GatewayConfig
from lensllm.db import init_db, get_connection

_db1 = make_temp_db()


def test_config_defaults():
    cfg = get_config()
    assert isinstance(cfg, GatewayConfig)
    assert cfg.similarity_threshold == 0.92
    assert cfg.rate_limit_rate == 2.0
    assert cfg.retries == 3
run_test("Phase 1", "Config dataclass has correct defaults", test_config_defaults)


def test_configure_updates_values():
    configure(similarity_threshold=0.85, retries=5)
    cfg = get_config()
    assert cfg.similarity_threshold == 0.85
    assert cfg.retries == 5
    configure(similarity_threshold=0.92, retries=3)  # reset
run_test("Phase 1", "configure() updates global config", test_configure_updates_values)


def test_configure_rejects_unknown_key():
    try:
        configure(nonexistent_key="value")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
run_test("Phase 1", "configure() rejects unknown keys", test_configure_rejects_unknown_key)


def test_db_schema_tables():
    init_db(_db1)
    conn = get_connection(_db1)
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    conn.close()
    assert "cache_entries"  in tables
    assert "request_logs"   in tables
run_test("Phase 1", "DB schema creates tables", test_db_schema_tables)


def test_db_wal_mode():
    conn = get_connection(_db1)
    mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    conn.close()
    assert mode == "wal"
run_test("Phase 1", "WAL journal mode enabled", test_db_wal_mode)


def test_db_foreign_keys():
    conn = get_connection(_db1)
    fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
    conn.close()
    assert fk == 1
run_test("Phase 1", "Foreign key enforcement ON", test_db_foreign_keys)


def test_db_init_idempotent():
    init_db(_db1)
    init_db(_db1)  # second call must not raise
    conn = get_connection(_db1)
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    conn.close()
    assert len(tables) >= 2
run_test("Phase 1", "init_db() is idempotent", test_db_init_idempotent)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Semantic Cache + Logging + Decorator
# ═══════════════════════════════════════════════════════════════════════════════

section("Phase 2 — Semantic Cache, Logging & @track()")

import numpy as np
from lensllm.cache.semantic import embed, cosine_similarity, embedding_to_bytes, bytes_to_embedding
from lensllm.cache.store import CacheStore
from lensllm.logging.store import LogStore, LogEntry
from lensllm import track

_db2 = make_temp_db()
init_db(_db2)
configure(db_path=_db2)

print(f"\n  {YELLOW}Loading sentence-transformer model (first run may take a moment)…{RESET}")


def test_embed_shape():
    v = embed("What is Python?")
    assert v.shape == (384,)
    assert v.dtype == np.float32
run_test("Phase 2", "embed() returns (384,) float32 vector", test_embed_shape)


def test_cosine_identical():
    v = embed("What is Python?")
    assert cosine_similarity(v, v) > 0.9999
run_test("Phase 2", "cosine_similarity(v, v) ≈ 1.0", test_cosine_identical)


def test_cosine_similar_prompts():
    v1 = embed("What is Python programming language?")
    v2 = embed("Can you tell me about Python as a programming language?")
    score = cosine_similarity(v1, v2)
    assert score > 0.88, f"Expected >0.88, got {score:.4f}"
run_test("Phase 2", "Similar prompts have cosine similarity > 0.88", test_cosine_similar_prompts)


def test_cosine_dissimilar_prompts():
    v1 = embed("What is Python programming language?")
    v2 = embed("What is the best chocolate cake recipe?")
    score = cosine_similarity(v1, v2)
    assert score < 0.75, f"Expected <0.75, got {score:.4f}"
run_test("Phase 2", "Dissimilar prompts have cosine similarity < 0.75", test_cosine_dissimilar_prompts)


def test_embedding_serialization():
    v = embed("Serialisation test")
    blob = embedding_to_bytes(v)
    v2   = bytes_to_embedding(blob)
    np.testing.assert_array_almost_equal(v, v2)
run_test("Phase 2", "Embedding BLOB roundtrip preserves values", test_embedding_serialization)


def test_cache_miss_then_hit():
    store  = CacheStore(db_path=_db2)
    prompt = "integration test prompt for cache"
    emb    = embed(prompt)

    # First lookup must miss
    miss = store.lookup(emb, threshold=0.92)
    assert miss is None, "Expected cache miss before write"

    # Write and look up again
    store.write(prompt, emb, "cached answer", ttl=None)
    hit = store.lookup(emb, threshold=0.92)
    assert hit is not None
    assert hit.response == "cached answer"
    assert hit.similarity > 0.999
run_test("Phase 2", "CacheStore: miss then write then hit", test_cache_miss_then_hit)


def test_cache_ttl_expiry():
    store  = CacheStore(db_path=_db2)
    prompt = "ttl test prompt"
    emb    = embed(prompt)
    store.write(prompt, emb, "will expire", ttl=1)
    assert store.lookup(emb, threshold=0.90) is not None
    time.sleep(1.2)
    assert store.lookup(emb, threshold=0.90) is None
run_test("Phase 2", "CacheStore: TTL expiry works", test_cache_ttl_expiry)


def test_logstore_miss_and_stats():
    import uuid
    ls = LogStore(db_path=_db2)
    ls.log_request(LogEntry(
        trace_id=str(uuid.uuid4()), span_id=str(uuid.uuid4()),
        function_name="test_fn", prompt="hello", response="world",
        cache_hit=False, latency_ms=250.0, cost_estimate=0.0005,
        timestamp=time.time()
    ))
    stats = ls.get_stats()
    assert stats["total_calls"] >= 1
    assert stats["total_cost_usd"] >= 0.0005
run_test("Phase 2", "LogStore: request persisted and stats correct", test_logstore_miss_and_stats)


def test_decorator_cache_hit_skips_fn():
    _db = make_temp_db(); init_db(_db); configure(db_path=_db)
    call_count = {"n": 0}

    @track(cache=True, similarity_threshold=0.99)
    def my_llm(prompt: str) -> str:
        call_count["n"] += 1
        return f"response::{prompt}"

    r1 = my_llm("What is Python?")
    r2 = my_llm("What is Python?")  # exact same → must hit cache
    assert r1 == r2
    assert call_count["n"] == 1, f"Expected 1 call, got {call_count['n']}"
run_test("Phase 2", "@track: cache hit skips wrapped function", test_decorator_cache_hit_skips_fn)


def test_decorator_cache_false_always_calls():
    _db = make_temp_db(); init_db(_db); configure(db_path=_db)
    call_count = {"n": 0}

    @track(cache=False)
    def my_llm(prompt: str) -> str:
        call_count["n"] += 1
        return "ok"

    my_llm("same prompt")
    my_llm("same prompt")
    assert call_count["n"] == 2
run_test("Phase 2", "@track(cache=False): always calls function", test_decorator_cache_false_always_calls)


def test_decorator_error_logged_and_reraised():
    _db = make_temp_db(); init_db(_db); configure(db_path=_db)

    @track(cache=False)
    def bad_fn(prompt: str) -> str:
        raise ValueError("deliberate error")

    try:
        bad_fn("test")
        assert False, "Should have raised"
    except ValueError:
        pass

    ls = LogStore(db_path=_db)
    logs = ls.get_recent_logs(limit=5)
    assert any(l["error"] is not None for l in logs)
run_test("Phase 2", "@track: exception logged and re-raised", test_decorator_error_logged_and_reraised)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Retry + Fallback Chain
# ═══════════════════════════════════════════════════════════════════════════════

section("Phase 3 — Retry with Backoff + Fallback Chain")

from unittest.mock import patch
from lensllm.retry import retry_with_backoff, compute_delay
from lensllm.fallback import FallbackChain


def test_retry_backoff_doubles():
    d0 = compute_delay(0, base_delay=1.0, max_delay=60.0, jitter=False)
    d1 = compute_delay(1, base_delay=1.0, max_delay=60.0, jitter=False)
    d2 = compute_delay(2, base_delay=1.0, max_delay=60.0, jitter=False)
    assert d0 == 1.0 and d1 == 2.0 and d2 == 4.0
run_test("Phase 3", "compute_delay: doubles each attempt without jitter", test_retry_backoff_doubles)


def test_retry_max_delay_cap():
    d = compute_delay(10, base_delay=1.0, max_delay=5.0, jitter=False)
    assert d == 5.0
run_test("Phase 3", "compute_delay: respects max_delay cap", test_retry_max_delay_cap)


def test_retry_jitter_in_bounds():
    for _ in range(50):
        d = compute_delay(2, base_delay=1.0, max_delay=60.0, jitter=True)
        assert 0.0 <= d <= 4.0, f"Jitter out of bounds: {d}"
run_test("Phase 3", "compute_delay: jitter stays within [0, computed] bounds", test_retry_jitter_in_bounds)


def test_retry_success_after_failures():
    calls = {"n": 0}
    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "success"

    with patch("lensllm.retry.time.sleep"):
        result = retry_with_backoff(
            flaky, args=("hello",), max_retries=3, jitter=False,
            retry_on=(ConnectionError,)
        )
    assert result == "success"
    assert calls["n"] == 3
run_test("Phase 3", "retry_with_backoff: succeeds after 2 transient failures", test_retry_success_after_failures)


def test_retry_exhausted_raises():
    def always_fail(prompt):
        raise TimeoutError("always")

    with patch("lensllm.retry.time.sleep"):
        try:
            retry_with_backoff(always_fail, args=("x",), max_retries=2,
                               retry_on=(TimeoutError,), jitter=False)
            assert False, "Should have raised"
        except TimeoutError:
            pass
run_test("Phase 3", "retry_with_backoff: raises after all retries exhausted", test_retry_exhausted_raises)


def test_retry_non_retryable_immediate():
    calls = {"n": 0}
    def fn(p):
        calls["n"] += 1
        raise ValueError("not retryable")

    with patch("lensllm.retry.time.sleep") as mock_sleep:
        try:
            retry_with_backoff(fn, args=("x",), max_retries=3,
                               retry_on=(ConnectionError,), jitter=False)
        except ValueError:
            pass
    assert calls["n"] == 1
    mock_sleep.assert_not_called()
run_test("Phase 3", "retry_with_backoff: non-retryable exception propagates immediately", test_retry_non_retryable_immediate)


def test_fallback_first_succeeds():
    p1 = lambda p: "p1_response"; p1.__name__ = "p1"
    p2 = lambda p: "p2_response"; p2.__name__ = "p2"
    result = FallbackChain([p1, p2]).run("q")
    assert result.succeeded
    assert result.response == "p1_response"
    assert result.succeeded_with == "p1"
run_test("Phase 3", "FallbackChain: returns first provider's response", test_fallback_first_succeeds)


def test_fallback_skips_failed_providers():
    def fail(p): raise ConnectionError("fail"); fail.__name__ = "fail"
    def ok(p):   return "from_ok";              ok.__name__   = "ok"
    result = FallbackChain([fail, ok]).run("q")
    assert result.succeeded
    assert result.response == "from_ok"
    assert result.succeeded_with == "ok"
    assert result.attempts[0][1] is not None  # fail recorded error
run_test("Phase 3", "FallbackChain: skips failed provider, uses next", test_fallback_skips_failed_providers)


def test_fallback_all_fail():
    def f1(p): raise ValueError("f1"); f1.__name__ = "f1"
    def f2(p): raise ValueError("f2"); f2.__name__ = "f2"
    result = FallbackChain([f1, f2]).run("q")
    assert not result.succeeded
    assert result.response is None
    assert len(result.attempts) == 2
run_test("Phase 3", "FallbackChain: returns failed result when all fail", test_fallback_all_fail)


def test_decorator_retries_on_transient_failure():
    _db = make_temp_db(); init_db(_db); configure(db_path=_db)
    calls = {"n": 0}

    @track(cache=False, retries=3, retry_on=(ConnectionError,), jitter=False, base_delay=0.0)
    def flaky_llm(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "recovered"

    with patch("lensllm.retry.time.sleep"):
        result = flaky_llm("test prompt")
    assert result == "recovered"
    assert calls["n"] == 3
run_test("Phase 3", "@track: retries on transient failure, returns response", test_decorator_retries_on_transient_failure)


def test_decorator_uses_fallback_on_primary_failure():
    _db = make_temp_db(); init_db(_db); configure(db_path=_db)

    def backup_llm(prompt: str) -> str:
        return "fallback_response"
    backup_llm.__name__ = "backup_llm"

    @track(cache=False, retries=1, retry_on=(RuntimeError,),
           fallback=[backup_llm], jitter=False, base_delay=0.0)
    def primary_llm(prompt: str) -> str:
        raise RuntimeError("primary down")

    with patch("lensllm.retry.time.sleep"):
        result = primary_llm("test")
    assert result == "fallback_response"
run_test("Phase 3", "@track: uses fallback when primary exhausts retries", test_decorator_uses_fallback_on_primary_failure)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Rate Limiter + Concurrency Queue
# ═══════════════════════════════════════════════════════════════════════════════

section("Phase 4 — Rate Limiter (Token Bucket) + Concurrency Queue")

from lensllm.rate_limiter import TokenBucketLimiter, get_limiter
from lensllm.queue import RequestQueue, get_queue


def test_bucket_starts_full():
    limiter = TokenBucketLimiter(rate=2.0, max_tokens=10.0)
    assert abs(limiter.available_tokens - 10.0) < 0.1
run_test("Phase 4", "TokenBucket: starts at max_tokens", test_bucket_starts_full)


def test_bucket_consumes_one_token():
    limiter = TokenBucketLimiter(rate=2.0, max_tokens=10.0)
    before = limiter.available_tokens
    limiter.acquire(block=False)
    assert abs((before - limiter.available_tokens) - 1.0) < 0.05
run_test("Phase 4", "TokenBucket: acquire() consumes exactly 1 token", test_bucket_consumes_one_token)


def test_bucket_non_blocking_false_when_empty():
    limiter = TokenBucketLimiter(rate=0.001, max_tokens=1.0)
    limiter.acquire(block=False)   # drain
    result = limiter.acquire(block=False)
    assert result is False
run_test("Phase 4", "TokenBucket: non-blocking acquire returns False when empty", test_bucket_non_blocking_false_when_empty)


def test_bucket_refills_over_time():
    limiter = TokenBucketLimiter(rate=10.0, max_tokens=10.0)
    for _ in range(10):
        limiter.acquire(block=False)     # drain
    time.sleep(0.5)                      # 0.5s × 10 tokens/s = ~5 tokens
    assert limiter.available_tokens == pytest.approx(5.0, abs=1.5)
run_test("Phase 4", "TokenBucket: refills at configured rate over time", test_bucket_refills_over_time)


def test_bucket_thread_safe():
    """15 threads racing non-blocking acquires against a 10-token bucket — at most 10 succeed."""
    limiter = TokenBucketLimiter(rate=0.001, max_tokens=10.0)
    successes = []
    lock = threading.Lock()
    barrier = threading.Barrier(15)

    def try_acquire():
        barrier.wait()
        r = limiter.acquire(block=False)
        with lock:
            successes.append(r)

    threads = [threading.Thread(target=try_acquire) for _ in range(15)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert sum(1 for r in successes if r) <= 10
run_test("Phase 4", "TokenBucket: thread-safe — at most max_tokens concurrent acquires", test_bucket_thread_safe)


def test_limiter_registry_shared():
    l1 = get_limiter(rate=2.0, max_tokens=10.0)
    l2 = get_limiter(rate=2.0, max_tokens=10.0)
    assert l1 is l2
run_test("Phase 4", "get_limiter: same params → same object (shared bucket)", test_limiter_registry_shared)


def test_queue_bounds_concurrency():
    """max_concurrency=2 → at most 2 threads inside slot simultaneously."""
    queue = RequestQueue(max_concurrency=2)
    peaks = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def task():
        with queue.slot():
            barrier.wait()
            with lock:
                peaks.append(queue.active_count)
            time.sleep(0.05)

    threads = [threading.Thread(target=task) for _ in range(2)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert max(peaks) == 2
run_test("Phase 4", "RequestQueue: active_count reaches exactly max_concurrency", test_queue_bounds_concurrency)


def test_queue_resets_to_zero():
    queue = RequestQueue(max_concurrency=3)
    def task():
        with queue.slot():
            time.sleep(0.02)

    threads = [threading.Thread(target=task) for _ in range(3)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert queue.active_count == 0
run_test("Phase 4", "RequestQueue: active_count resets to 0 after completion", test_queue_resets_to_zero)


def test_queue_registry_shared():
    q1 = get_queue(5)
    q2 = get_queue(5)
    assert q1 is q2
run_test("Phase 4", "get_queue: same value → same object (shared queue)", test_queue_registry_shared)


def test_decorator_with_rate_limit_and_concurrency():
    """@track with rate_limit + max_concurrency completes without errors."""
    _db = make_temp_db(); init_db(_db); configure(db_path=_db)

    @track(
        cache=False,
        rate_limit_rate=100.0,   # very high rate so test isn't slow
        rate_limit_tokens=10.0,
        max_concurrency=3,
    )
    def fn(prompt: str) -> str:
        return f"ok::{prompt}"

    results = []
    def call():
        results.append(fn("test prompt"))

    threads = [threading.Thread(target=call) for _ in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert all(r == "ok::test prompt" for r in results)
    assert len(results) == 5
run_test("Phase 4", "@track with rate_limit + max_concurrency: 5 concurrent calls succeed", test_decorator_with_rate_limit_and_concurrency)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════



print(f"\n{BOLD}{'═' * 60}{RESET}")
print(f"{BOLD}  COMPLETE TEST SUMMARY{RESET}")
print(f"{'═' * 60}")

groups: dict[str, list[tuple[str, bool, str]]] = {}
for group, name, passed, detail in _results:
    groups.setdefault(group, []).append((name, passed, detail))

total_pass = total_fail = 0
for group, tests in groups.items():
    p = sum(1 for _, ok, _ in tests if ok)
    f = sum(1 for _, ok, _ in tests if not ok)
    total_pass += p
    total_fail += f
    status = f"{GREEN}{p}/{len(tests)} PASSED{RESET}" if f == 0 else f"{RED}{p}/{len(tests)} PASSED, {f} FAILED{RESET}"
    print(f"  {BOLD}{group:<40}{RESET} {status}")

print(f"\n{'─' * 60}")
all_pass = total_fail == 0
overall = f"{GREEN}{BOLD}ALL {total_pass} TESTS PASSED ✅{RESET}" if all_pass else \
          f"{RED}{BOLD}{total_pass} passed, {total_fail} FAILED ❌{RESET}"
print(f"  {overall}")
print(f"{'─' * 60}\n")

# Cleanup temp DBs
import os
for db in [_db1, _db2]:
    try:
        os.unlink(db)
    except Exception:
        pass

sys.exit(0 if all_pass else 1)
