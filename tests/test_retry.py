"""
tests/test_retry.py
===================
Phase 3 tests: retry logic — backoff, jitter, retry_on, non-retryable errors.

Per skill.md rule #5: tests must pass before any refactoring touches retry.py.

We mock time.sleep so tests run fast and we can verify delay values precisely.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, call, MagicMock

from gateway_sdk.retry import retry_with_backoff, compute_delay, DEFAULT_RETRY_ON


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_failing_fn(fail_times: int, exc_type=ConnectionError, success_value="ok"):
    """Return a callable that fails `fail_times` times then returns `success_value`."""
    calls = {"n": 0}

    def fn(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise exc_type(f"failure #{calls['n']}")
        return success_value

    fn.calls = calls
    return fn


# ── compute_delay ─────────────────────────────────────────────────────────────

class TestComputeDelay:
    def test_no_jitter_doubles_each_attempt(self):
        assert compute_delay(0, base_delay=1.0, max_delay=60.0, jitter=False) == 1.0
        assert compute_delay(1, base_delay=1.0, max_delay=60.0, jitter=False) == 2.0
        assert compute_delay(2, base_delay=1.0, max_delay=60.0, jitter=False) == 4.0
        assert compute_delay(3, base_delay=1.0, max_delay=60.0, jitter=False) == 8.0

    def test_max_delay_caps_exponential_growth(self):
        delay = compute_delay(10, base_delay=1.0, max_delay=5.0, jitter=False)
        assert delay == 5.0

    def test_jitter_stays_within_bounds(self):
        for _ in range(100):
            delay = compute_delay(2, base_delay=1.0, max_delay=60.0, jitter=True)
            assert 0.0 <= delay <= 4.0  # attempt=2 → computed=4s, jitter ∈ [0, 4]

    def test_jitter_false_is_deterministic(self):
        d1 = compute_delay(1, base_delay=1.0, max_delay=60.0, jitter=False)
        d2 = compute_delay(1, base_delay=1.0, max_delay=60.0, jitter=False)
        assert d1 == d2 == 2.0


# ── retry_with_backoff ────────────────────────────────────────────────────────

class TestRetryWithBackoff:

    def test_success_on_first_try(self):
        """No retries needed if fn succeeds immediately."""
        with patch("gateway_sdk.retry.time.sleep") as mock_sleep:
            fn = make_failing_fn(0, success_value="result")
            result = retry_with_backoff(fn, max_retries=3, jitter=False)
        assert result == "result"
        assert fn.calls["n"] == 1
        mock_sleep.assert_not_called()

    def test_success_after_one_retry(self):
        """fn fails once then succeeds — should succeed with 1 retry."""
        with patch("gateway_sdk.retry.time.sleep"):
            fn = make_failing_fn(1, exc_type=ConnectionError, success_value="ok")
            result = retry_with_backoff(
                fn, max_retries=3, base_delay=1.0, jitter=False,
                retry_on=(ConnectionError,)
            )
        assert result == "ok"
        assert fn.calls["n"] == 2  # 1 failure + 1 success

    def test_success_after_max_retries(self):
        """fn fails exactly max_retries times then succeeds."""
        with patch("gateway_sdk.retry.time.sleep"):
            fn = make_failing_fn(3, exc_type=ConnectionError, success_value="ok")
            result = retry_with_backoff(
                fn, max_retries=3, base_delay=1.0, jitter=False,
                retry_on=(ConnectionError,)
            )
        assert result == "ok"
        assert fn.calls["n"] == 4  # 3 failures + 1 success

    def test_raises_after_all_retries_exhausted(self):
        """fn fails more times than max_retries — last exception must be raised."""
        with patch("gateway_sdk.retry.time.sleep"):
            fn = make_failing_fn(10, exc_type=ConnectionError)
            with pytest.raises(ConnectionError, match="failure #3"):
                # max_retries=2 → 1 initial call + 2 retries = 3 total calls
                # The last exception is from attempt #3
                retry_with_backoff(
                    fn, max_retries=2, jitter=False, retry_on=(ConnectionError,)
                )
            assert fn.calls["n"] == 3  # 1 initial + 2 retries

    def test_sleep_called_with_correct_delays(self):
        """Verify exponential backoff delays without jitter."""
        with patch("gateway_sdk.retry.time.sleep") as mock_sleep:
            fn = make_failing_fn(3, exc_type=ConnectionError, success_value="ok")
            retry_with_backoff(
                fn, max_retries=3, base_delay=1.0, max_delay=60.0, jitter=False,
                retry_on=(ConnectionError,)
            )
        # Attempt 0 failure → sleep 1s; attempt 1 → sleep 2s; attempt 2 → sleep 4s
        assert mock_sleep.call_count == 3
        mock_sleep.assert_any_call(pytest.approx(1.0))
        mock_sleep.assert_any_call(pytest.approx(2.0))
        mock_sleep.assert_any_call(pytest.approx(4.0))

    def test_non_retryable_exception_raises_immediately(self):
        """An exception NOT in retry_on must propagate without sleeping."""
        with patch("gateway_sdk.retry.time.sleep") as mock_sleep:
            fn = make_failing_fn(1, exc_type=ValueError)
            with pytest.raises(ValueError):
                retry_with_backoff(
                    fn, max_retries=3, retry_on=(ConnectionError,)
                )
        assert fn.calls["n"] == 1  # called only once, no retry
        mock_sleep.assert_not_called()

    def test_custom_retry_on(self):
        """retry_on parameter should determine which exceptions trigger retry."""
        with patch("gateway_sdk.retry.time.sleep"):
            fn = make_failing_fn(2, exc_type=TimeoutError, success_value="done")
            result = retry_with_backoff(
                fn, max_retries=3, retry_on=(TimeoutError,), jitter=False
            )
        assert result == "done"

    def test_args_and_kwargs_forwarded(self):
        """Positional and keyword args must be forwarded to fn on every attempt."""
        call_log = []

        def fn(prompt, temperature=0.7):
            call_log.append((prompt, temperature))
            if len(call_log) < 2:
                raise ConnectionError("first fail")
            return "response"

        with patch("gateway_sdk.retry.time.sleep"):
            result = retry_with_backoff(
                fn, args=("hello",), kwargs={"temperature": 0.5},
                max_retries=2, jitter=False, retry_on=(ConnectionError,)
            )

        assert result == "response"
        assert all(p == "hello" and t == 0.5 for p, t in call_log)

    def test_max_delay_respected(self):
        """Computed delay must never exceed max_delay."""
        with patch("gateway_sdk.retry.time.sleep") as mock_sleep:
            fn = make_failing_fn(5, exc_type=ConnectionError, success_value="ok")
            retry_with_backoff(
                fn, max_retries=5, base_delay=10.0, max_delay=15.0, jitter=False,
                retry_on=(ConnectionError,)
            )
        for sleep_call in mock_sleep.call_args_list:
            assert sleep_call.args[0] <= 15.0
