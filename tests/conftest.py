"""
tests/conftest.py
=================
Shared pytest fixtures for the undertow_llm test suite.

Fixtures available to all tests (no import needed — pytest discovers conftest.py):
    temp_db        — in-memory SQLite DB path (isolated per test)
    mock_llm       — a simple function that fakes an LLM response
    sample_prompts — a list of test prompts including near-duplicates
"""

import pytest
import tempfile
import os

from undertow_llm.db import init_db
from undertow_llm.config import configure


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """
    Create a fresh, isolated SQLite database for each test.

    Uses pytest's built-in tmp_path fixture, which creates a unique temporary
    directory per test — no test can pollute another's database.
    """
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.delenv("UNDERTOW_LLM_POSTGRES_URL", raising=False)
    db_path = str(tmp_path / "test_undertow_llm.db")
    from undertow_llm.backends.factory import reset_backends
    reset_backends()
    # Point global config at this temp DB
    configure(db_path=db_path, postgres_url=None)
    # Initialise schema
    init_db(db_path)
    yield db_path
    reset_backends()


@pytest.fixture
def mock_llm():
    """
    A callable that mimics an LLM function.
    Returns a deterministic response based on the prompt for reproducible tests.
    Tracks call count so tests can verify cache-hit behaviour (function not called
    a second time for cached prompts).
    """
    call_count = {"n": 0}

    def _llm(prompt: str) -> str:
        call_count["n"] += 1
        return f"Mock response for: {prompt}"

    _llm.call_count = call_count
    return _llm


@pytest.fixture
def sample_prompts():
    """
    A curated set of prompts that includes:
      - Semantically similar pairs (should produce cache hits at threshold=0.92)
      - Semantically dissimilar prompts (should produce cache misses)
    """
    return {
        "original": "What is Python programming language?",
        "near_duplicate": "Can you tell me about Python as a programming language?",
        "dissimilar": "What is the best recipe for chocolate cake?",
    }
