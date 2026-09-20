"""
tests/test_decorator.py
=======================
Phase 1 tests: decorator structure, passthrough behaviour, import health.
Phase 2 tests (cache + logging) will be added when Phase 2 is implemented.
"""

import pytest
from undertow_llm import track
from undertow_llm.config import configure, get_config


class TestDecoratorPhase1:
    """Basic decorator correctness — Phase 1 (passthrough stub)."""

    def test_decorated_function_returns_correct_value(self, temp_db):
        """The decorator must not alter the wrapped function's return value."""
        @track(cache=False)
        def call_llm(prompt: str) -> str:
            return f"Response: {prompt}"

        result = call_llm("Hello!")
        assert result == "Response: Hello!"

    def test_decorated_function_is_callable(self, temp_db):
        """Decorated function should be callable just like the original."""
        @track()
        def my_llm(prompt: str) -> str:
            return "ok"

        assert callable(my_llm)

    def test_decorator_preserves_function_name(self, temp_db):
        """functools.wraps must preserve __name__ and __doc__."""
        @track()
        def call_gemini(prompt: str) -> str:
            """Calls Gemini."""
            return "gemini response"

        assert call_gemini.__name__ == "call_gemini"
        assert call_gemini.__doc__ == "Calls Gemini."

    def test_decorator_marks_tracked_function(self, temp_db):
        """Decorated functions get a _gateway_tracked marker for introspection."""
        @track()
        def my_fn(prompt: str) -> str:
            return "x"

        assert getattr(my_fn, "_gateway_tracked", False) is True

    def test_decorator_accepts_all_phase1_params(self, temp_db):
        """All Phase 1 parameters should be accepted without error."""
        @track(cache=True, similarity_threshold=0.88, cache_ttl=300)
        def fn(prompt: str) -> str:
            return "ok"

        assert fn("test") == "ok"

    def test_decorator_passthrough_with_extra_args(self, temp_db):
        """Extra positional and keyword args must be forwarded to the wrapped fn."""
        @track()
        def fn(prompt: str, temperature: float = 0.7, stream: bool = False) -> dict:
            return {"prompt": prompt, "temperature": temperature, "stream": stream}

        result = fn("hello", temperature=0.5, stream=True)
        assert result == {"prompt": "hello", "temperature": 0.5, "stream": True}


class TestConfig:
    """Global configuration tests."""

    def test_configure_updates_global_config(self, temp_db):
        configure(similarity_threshold=0.85, retries=5)
        cfg = get_config()
        assert cfg.similarity_threshold == 0.85
        assert cfg.retries == 5

    def test_configure_rejects_unknown_keys(self, temp_db):
        with pytest.raises(ValueError, match="Unknown undertow_llm config key"):
            configure(nonexistent_key="value")

    def test_per_call_threshold_overrides_global(self, temp_db):
        """Per-call similarity_threshold overrides the global config."""
        configure(similarity_threshold=0.99)

        @track(similarity_threshold=0.70)
        def fn(prompt: str) -> str:
            return "ok"

        assert fn("test") == "ok"


# ── Phase 2: Cache + Logging Integration ────────────────────────────────────

class TestDecoratorPhase2:
    """Test the full cache + logging pipeline via the decorator."""

    def test_cache_miss_calls_function(self, temp_db, mock_llm):
        """On a cache miss the wrapped function must be called exactly once."""
        tracked = track(cache=True)(mock_llm)
        result = tracked("What is Python?")
        assert result == "Mock response for: What is Python?"
        assert mock_llm.call_count["n"] == 1

    def test_cache_hit_skips_function(self, temp_db, mock_llm):
        """
        On the second call with an identical prompt the wrapped function
        must NOT be called again — the cache must serve the response.
        """
        tracked = track(cache=True, similarity_threshold=0.99)(mock_llm)
        r1 = tracked("What is Python?")
        r2 = tracked("What is Python?")   # identical → should hit cache

        assert r1 == r2
        assert mock_llm.call_count["n"] == 1  # function called only once

    def test_cache_false_always_calls_function(self, temp_db, mock_llm):
        """cache=False must bypass the cache and call the function every time."""
        tracked = track(cache=False)(mock_llm)
        tracked("What is Python?")
        tracked("What is Python?")   # same prompt, cache disabled
        assert mock_llm.call_count["n"] == 2

    def test_response_returned_unmodified(self, temp_db):
        """The decorator must not alter the wrapped function's return value."""
        @track(cache=True)
        def fn(prompt: str) -> str:
            return f"answer::{prompt}"

        assert fn("hello") == "answer::hello"

    def test_log_written_on_miss(self, temp_db, mock_llm):
        """A request_logs entry must be created for every miss."""
        from undertow_llm.logging.store import LogStore
        tracked = track(cache=True)(mock_llm)
        tracked("Python question")

        store = LogStore(db_path=temp_db)
        logs = store.get_recent_logs(limit=5)
        assert len(logs) == 1
        assert logs[0]["cache_hit"] == 0

    def test_log_written_on_hit(self, temp_db, mock_llm):
        """A request_logs entry must be created for cache hits too."""
        from undertow_llm.logging.store import LogStore
        tracked = track(cache=True, similarity_threshold=0.99)(mock_llm)
        tracked("Python question")  # miss
        tracked("Python question")  # hit

        store = LogStore(db_path=temp_db)
        logs = store.get_recent_logs(limit=5)
        assert len(logs) == 2
        # Newest first — the hit should be the first row
        assert logs[0]["cache_hit"] == 1
        assert logs[1]["cache_hit"] == 0

    def test_exception_is_reraised(self, temp_db):
        """Errors from the wrapped function must propagate to the caller."""
        @track(cache=True)
        def fn(prompt: str) -> str:
            raise ValueError("LLM failed")

        with pytest.raises(ValueError, match="LLM failed"):
            fn("test")

    def test_cost_per_call_override(self, temp_db):
        """cost_per_call parameter should override auto token-estimate cost."""
        from undertow_llm.logging.store import LogStore
        @track(cache=False, cost_per_call=0.0015)
        def custom_priced_fn(prompt: str) -> str:
            return "custom priced response"

        custom_priced_fn("Calculate price per call for 1M tokens")

        store = LogStore(db_path=temp_db)
        logs = store.get_recent_logs(limit=1)
        assert len(logs) == 1
        assert logs[0]["cost_estimate"] == 0.0015
