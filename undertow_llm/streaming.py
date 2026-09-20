"""
undertow_llm.streaming
=====================
Generator/stream passthrough support for @track().

When an LLM function returns a generator (streaming response), this module
wraps it so:
  - Each token/chunk is yielded unchanged to the caller
  - After the stream is exhausted, on_complete(full_text, latency_ms) is called
    to trigger logging and cache writes (same as the non-streaming path)

Supports:
  - Sync generators   (types.GeneratorType)
  - Async generators  (types.AsyncGeneratorType)

Design note — why not buffer everything?
    Buffering the full response before yielding defeats the purpose of streaming
    (latency-to-first-token). Instead we accumulate chunks in a list and join
    them only when the generator is exhausted.
"""

from __future__ import annotations

import time
import types
from typing import Any, Callable, Generator, AsyncGenerator


def is_generator(obj: Any) -> bool:
    """Return True if obj is a sync or async generator."""
    return isinstance(obj, (types.GeneratorType, types.AsyncGeneratorType))


def wrap_stream(
    gen: Generator,
    on_complete: Callable[[str, float], None],
    start_time: float,
) -> Generator:
    """
    Wrap a sync generator so that:
      - Each chunk is yielded to the caller unchanged
      - After exhaustion, on_complete(full_text, latency_ms) is called

    Args:
        gen:          The sync generator returned by the LLM function
        on_complete:  Callback(full_text, latency_ms) fired when stream ends
        start_time:   perf_counter() timestamp of when the call started

    Yields:
        Each chunk from gen, unchanged.
    """
    chunks: list[str] = []
    try:
        for chunk in gen:
            chunks.append(str(chunk) if chunk is not None else "")
            yield chunk
    finally:
        # Always called: normal exhaustion OR caller breaking out of loop
        latency_ms = (time.perf_counter() - start_time) * 1000
        full_text = "".join(chunks)
        on_complete(full_text, latency_ms)


async def wrap_async_stream(
    gen: AsyncGenerator,
    on_complete: Callable[[str, float], None],
    start_time: float,
) -> AsyncGenerator:
    """
    Wrap an async generator so that:
      - Each chunk is yielded to the caller unchanged
      - After exhaustion, on_complete(full_text, latency_ms) is called

    Args:
        gen:          The async generator returned by the LLM function
        on_complete:  Callback(full_text, latency_ms) fired when stream ends
        start_time:   perf_counter() timestamp of when the call started

    Yields:
        Each chunk from gen, unchanged.
    """
    chunks: list[str] = []
    try:
        async for chunk in gen:
            chunks.append(str(chunk) if chunk is not None else "")
            yield chunk
    finally:
        latency_ms = (time.perf_counter() - start_time) * 1000
        full_text = "".join(chunks)
        on_complete(full_text, latency_ms)
