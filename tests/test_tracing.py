"""
tests/test_tracing.py
=====================
Unit tests for undertow_llm.tracing (Phase 7).
"""
import threading
import pytest
from undertow_llm.tracing import (
    get_current_trace,
    set_current_trace,
    reset_current_trace,
)


def test_no_trace_by_default():
    """Before any set_current_trace(), get_current_trace() returns None."""
    assert get_current_trace() is None


def test_set_and_get():
    """After set_current_trace(), get_current_trace() returns the right values."""
    token = set_current_trace("trace-abc", "span-123")
    try:
        result = get_current_trace()
        assert result == ("trace-abc", "span-123")
    finally:
        reset_current_trace(token)


def test_reset_restores_none():
    """After reset, context returns to None."""
    token = set_current_trace("t1", "s1")
    reset_current_trace(token)
    assert get_current_trace() is None


def test_nested_contexts():
    """Inner set_current_trace should shadow outer; reset restores outer."""
    outer_token = set_current_trace("outer-trace", "outer-span")
    try:
        assert get_current_trace() == ("outer-trace", "outer-span")

        inner_token = set_current_trace("inner-trace", "inner-span")
        try:
            assert get_current_trace() == ("inner-trace", "inner-span")
        finally:
            reset_current_trace(inner_token)

        # After inner reset, outer context should be visible again
        assert get_current_trace() == ("outer-trace", "outer-span")
    finally:
        reset_current_trace(outer_token)

    assert get_current_trace() is None


def test_independent_threads():
    """Threads must have independent trace contexts (no cross-contamination)."""
    results = {}

    def thread_fn(thread_id, trace_id, span_id):
        token = set_current_trace(trace_id, span_id)
        # Simulate some work
        import time; time.sleep(0.01)
        results[thread_id] = get_current_trace()
        reset_current_trace(token)

    threads = [
        threading.Thread(target=thread_fn, args=(i, f"trace-{i}", f"span-{i}"))
        for i in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i in range(5):
        assert results[i] == (f"trace-{i}", f"span-{i}")


def test_multiple_set_same_trace():
    """Setting trace multiple times accumulates correctly via token stack."""
    t1 = set_current_trace("A", "span1")
    t2 = set_current_trace("A", "span2")
    assert get_current_trace() == ("A", "span2")
    reset_current_trace(t2)
    assert get_current_trace() == ("A", "span1")
    reset_current_trace(t1)
    assert get_current_trace() is None
