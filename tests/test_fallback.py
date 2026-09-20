"""
tests/test_fallback.py
======================
Phase 3 tests: FallbackChain behaviour.
"""

from __future__ import annotations

import pytest
from undertow_llm.fallback import FallbackChain, FallbackResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_provider(name: str, *, fails: bool = False, response: str = "ok"):
    def fn(prompt, *args, **kwargs):
        if fails:
            raise ConnectionError(f"{name} failed")
        return response
    fn.__name__ = name
    return fn


# ── FallbackChain ─────────────────────────────────────────────────────────────

class TestFallbackChain:

    def test_empty_providers_raises(self):
        with pytest.raises(ValueError, match="at least one provider"):
            FallbackChain([])

    def test_first_provider_succeeds(self):
        """If first provider works, return its response without trying others."""
        p1 = make_provider("p1", response="from_p1")
        p2 = make_provider("p2", response="from_p2")

        chain = FallbackChain([p1, p2])
        result = chain.run("hello")

        assert result.succeeded is True
        assert result.response == "from_p1"
        assert result.succeeded_with == "p1"
        assert len(result.attempts) == 1
        assert result.attempts[0] == ("p1", None)

    def test_falls_back_to_second_on_first_failure(self):
        """If first fails, must try second and return its response."""
        p1 = make_provider("p1", fails=True)
        p2 = make_provider("p2", response="fallback_response")

        chain = FallbackChain([p1, p2])
        result = chain.run("prompt")

        assert result.succeeded is True
        assert result.response == "fallback_response"
        assert result.succeeded_with == "p2"
        # First attempt recorded error, second recorded success
        assert result.attempts[0][0] == "p1"
        assert result.attempts[0][1] is not None   # has error string
        assert result.attempts[1] == ("p2", None)  # success

    def test_returns_failed_result_when_all_fail(self):
        """If all providers fail, result.succeeded must be False and response None."""
        p1 = make_provider("p1", fails=True)
        p2 = make_provider("p2", fails=True)
        p3 = make_provider("p3", fails=True)

        chain = FallbackChain([p1, p2, p3])
        result = chain.run("prompt")

        assert result.succeeded is False
        assert result.response is None
        assert result.succeeded_with is None
        assert len(result.attempts) == 3
        # All attempts must have error strings
        assert all(err is not None for _, err in result.attempts)

    def test_attempts_logged_in_order(self):
        """Attempt trail must match the order providers were tried."""
        p1 = make_provider("alpha", fails=True)
        p2 = make_provider("beta",  fails=True)
        p3 = make_provider("gamma", response="ok")

        chain = FallbackChain([p1, p2, p3])
        result = chain.run("test")

        names = [name for name, _ in result.attempts]
        assert names == ["alpha", "beta", "gamma"]

    def test_prompt_forwarded_to_each_provider(self):
        """The same prompt must be passed to every provider tried."""
        received = []

        def p1(prompt):
            received.append(("p1", prompt))
            raise ConnectionError()

        def p2(prompt):
            received.append(("p2", prompt))
            return "ok"

        p1.__name__ = "p1"
        p2.__name__ = "p2"

        chain = FallbackChain([p1, p2])
        chain.run("my prompt")

        assert received == [("p1", "my prompt"), ("p2", "my prompt")]

    def test_total_latency_is_positive(self):
        """total_latency_ms must be a positive float."""
        p = make_provider("p", response="ok")
        chain = FallbackChain([p])
        result = chain.run("test")
        assert result.total_latency_ms >= 0.0

    def test_single_provider_success(self):
        """FallbackChain with one provider that succeeds."""
        p = make_provider("solo", response="solo_response")
        result = FallbackChain([p]).run("q")
        assert result.response == "solo_response"
        assert result.succeeded_with == "solo"
