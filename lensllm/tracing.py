"""
lensllm.tracing
===================
Distributed trace propagation across nested @track() calls.

Uses Python's contextvars.ContextVar so trace_id propagates automatically
across threads and async tasks without any user-visible API change.

How it works:
    - First @track() call in a chain: generates new trace_id, sets ContextVar
    - Nested @track() calls (agent loops, RAG pipelines): read parent trace_id
      from ContextVar, reuse it, and set their span's parent_span_id
    - ContextVar is reset after each wrapper() returns (using Token.reset())
      so sibling calls at the same nesting level get independent spans

Design note — ContextVar vs threading.local:
    ContextVar is the modern Python 3.7+ approach. It works correctly with
    asyncio (each Task gets its own copy), threading, and concurrent.futures.
    threading.local would fail for async callers.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Optional

# (trace_id, span_id) of the currently-executing @track wrapper, or None
_current_trace: ContextVar[Optional[tuple[str, str]]] = ContextVar(
    "_gateway_current_trace", default=None
)


def get_current_trace() -> Optional[tuple[str, str]]:
    """
    Return (trace_id, span_id) of the innermost active @track call, or None.

    Called at the start of wrapper() to detect whether we're inside a parent span.
    """
    return _current_trace.get()


def set_current_trace(trace_id: str, span_id: str) -> Token:
    """
    Set the current trace context.  Returns a Token to restore the previous
    value when the wrapper exits.

    Usage:
        token = set_current_trace(trace_id, span_id)
        try:
            ...  # the LLM call
        finally:
            reset_current_trace(token)
    """
    return _current_trace.set((trace_id, span_id))


def reset_current_trace(token: Token) -> None:
    """Restore the trace context to its value before set_current_trace()."""
    _current_trace.reset(token)
