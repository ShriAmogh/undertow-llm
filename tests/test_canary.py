"""
tests/test_canary.py
====================
Unit tests for lensllm.canary (Phase 9).
"""
import pytest
from lensllm.canary import parse_canary, should_use_canary, CanaryConfig


# ── should_use_canary ─────────────────────────────────────────────────────────

def test_weight_zero_never_canary():
    results = [should_use_canary(0.0) for _ in range(100)]
    assert all(r is False for r in results)

def test_weight_one_always_canary():
    results = [should_use_canary(1.0) for _ in range(100)]
    assert all(r is True for r in results)

def test_weight_half_probabilistic():
    """With weight=0.5, roughly half should be True (within tolerance)."""
    results = [should_use_canary(0.5) for _ in range(1000)]
    true_count = sum(results)
    # Allow wide tolerance: 300-700 out of 1000
    assert 300 <= true_count <= 700


# ── parse_canary ──────────────────────────────────────────────────────────────

def test_parse_canary_none():
    assert parse_canary(None) is None

def test_parse_canary_empty_dict():
    assert parse_canary({}) is None

def test_parse_canary_valid():
    fn = lambda p: "resp"
    cfg = parse_canary({"fn": fn, "weight": 0.1})
    assert isinstance(cfg, CanaryConfig)
    assert cfg.fn is fn
    assert cfg.weight == 0.1

def test_parse_canary_default_weight():
    fn = lambda p: "resp"
    cfg = parse_canary({"fn": fn})
    assert cfg.weight == 0.0

def test_parse_canary_invalid_weight_high():
    fn = lambda p: "resp"
    with pytest.raises(ValueError, match="weight"):
        parse_canary({"fn": fn, "weight": 1.5})

def test_parse_canary_invalid_weight_negative():
    fn = lambda p: "resp"
    with pytest.raises(ValueError, match="weight"):
        parse_canary({"fn": fn, "weight": -0.1})

def test_parse_canary_missing_fn():
    with pytest.raises(ValueError, match="fn"):
        parse_canary({"weight": 0.5})

def test_parse_canary_non_callable_fn():
    with pytest.raises(ValueError, match="fn"):
        parse_canary({"fn": "not_a_callable", "weight": 0.5})


# ── Decorator integration: canary routing ─────────────────────────────────────

def test_decorator_always_routes_to_canary(temp_db):
    """With weight=1.0 canary, the canary function must be called instead of primary."""
    from lensllm.decorator import track

    primary_calls = []
    canary_calls = []

    @track(cache=False, retries=0, canary={"fn": lambda p: canary_calls.append(p) or "canary!", "weight": 1.0})
    def primary(prompt):
        primary_calls.append(prompt)
        return "primary!"

    result = primary("test prompt")
    assert result == "canary!"
    assert len(primary_calls) == 0
    assert len(canary_calls) == 1


def test_decorator_never_routes_to_canary(temp_db):
    """With weight=0.0, primary function must always be called."""
    from lensllm.decorator import track

    canary_calls = []

    @track(cache=False, retries=0, canary={"fn": lambda p: canary_calls.append(p) or "canary!", "weight": 0.0})
    def primary(prompt):
        return "primary!"

    result = primary("test prompt")
    assert result == "primary!"
    assert len(canary_calls) == 0


def test_variant_tagging_in_log(temp_db):
    """Canary calls must have variant='canary' in request_logs."""
    from lensllm.decorator import track
    from lensllm.logging.store import LogStore

    @track(cache=False, retries=0, canary={"fn": lambda p: "canary_resp", "weight": 1.0})
    def primary(prompt):
        return "primary_resp"

    primary("hello")

    log_store = LogStore(temp_db)
    logs = log_store.get_recent_logs(limit=5)
    variants = [l["variant"] for l in logs if l["variant"] is not None]
    assert "canary" in variants
