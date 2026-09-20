"""
tests/test_streaming.py
=======================
Unit tests for gateway_sdk.streaming (Phase 5).
"""
import time
import pytest
from gateway_sdk.streaming import is_generator, wrap_stream


# ── is_generator ─────────────────────────────────────────────────────────────

def test_is_generator_true():
    def gen():
        yield 1
    assert is_generator(gen()) is True

def test_is_generator_false_string():
    assert is_generator("hello") is False

def test_is_generator_false_list():
    assert is_generator([1, 2, 3]) is False

def test_is_generator_false_none():
    assert is_generator(None) is False


# ── wrap_stream ───────────────────────────────────────────────────────────────

def test_wrap_stream_yields_all_tokens():
    """All tokens from the underlying generator must come through unchanged."""
    tokens = ["Hello", " ", "World", "!"]

    def source():
        yield from tokens

    collected = []
    callback_called = []

    def on_complete(full_text, latency_ms):
        callback_called.append((full_text, latency_ms))

    for chunk in wrap_stream(source(), on_complete, time.perf_counter()):
        collected.append(chunk)

    assert collected == tokens


def test_wrap_stream_calls_on_complete():
    """on_complete must be called exactly once after all tokens are consumed."""
    tokens = ["a", "b", "c"]

    def source():
        yield from tokens

    results = []

    def on_complete(full_text, latency_ms):
        results.append(full_text)

    list(wrap_stream(source(), on_complete, time.perf_counter()))

    assert len(results) == 1
    assert results[0] == "abc"


def test_wrap_stream_full_text_joined():
    """on_complete receives all chunks joined into a single string."""
    def source():
        yield "The "
        yield "quick "
        yield "brown fox"

    received = []

    def on_complete(full_text, latency_ms):
        received.append(full_text)

    list(wrap_stream(source(), on_complete, time.perf_counter()))
    assert received[0] == "The quick brown fox"


def test_wrap_stream_latency_positive():
    """on_complete latency_ms must be a non-negative float."""
    def source():
        yield "hi"

    latencies = []

    def on_complete(full_text, latency_ms):
        latencies.append(latency_ms)

    list(wrap_stream(source(), on_complete, time.perf_counter()))
    assert latencies[0] >= 0.0


def test_wrap_stream_on_complete_called_on_early_break():
    """on_complete must fire even when caller breaks early from the loop."""
    def source():
        yield "a"
        yield "b"
        yield "c"

    results = []

    def on_complete(full_text, latency_ms):
        results.append(full_text)

    gen = wrap_stream(source(), on_complete, time.perf_counter())
    next(gen)   # consume only first token then abandon
    gen.close()

    assert len(results) == 1
    assert results[0] == "a"


def test_wrap_stream_empty_generator():
    """An empty generator must call on_complete with empty string."""
    def source():
        return
        yield

    results = []

    def on_complete(full_text, latency_ms):
        results.append(full_text)

    list(wrap_stream(source(), on_complete, time.perf_counter()))
    assert results[0] == ""
