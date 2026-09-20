"""
tests/test_cache.py
===================
Phase 2 tests: semantic cache — embedding, cosine similarity, store.

We mock the sentence-transformer model so tests run fast (no 80MB model
download in CI), but keep the cosine similarity math unpatched so we
validate the algorithm itself.
"""

from __future__ import annotations

import time
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from undertow_llm.cache.semantic import cosine_similarity, embedding_to_bytes, bytes_to_embedding
from undertow_llm.cache.store import CacheStore


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cache_store(temp_db):
    return CacheStore(db_path=temp_db)


def _fake_embed(text: str) -> np.ndarray:
    """
    Deterministic fake embedding for tests.
    Near-duplicates (same first 20 chars) get a nearly identical vector.
    Dissimilar texts get a near-orthogonal vector.
    """
    np.random.seed(hash(text[:20]) % (2**31))
    base = np.random.rand(384).astype(np.float32)
    # Add a small perturbation for texts with same prefix (simulates near-duplicates)
    if len(text) > 20:
        np.random.seed(hash(text) % (2**31))
        perturbation = np.random.rand(384).astype(np.float32) * 0.05
        base = base + perturbation
    # Normalise
    return base / np.linalg.norm(base)


# ── Cosine Similarity ─────────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_score_zero(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_score_minus_one(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_similar_vectors_score_high(self):
        # Two nearly-identical unit vectors
        a = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        b = np.array([0.601, 0.799, 0.01], dtype=np.float32)
        b = b / np.linalg.norm(b)
        score = cosine_similarity(a, b)
        assert score > 0.99


# ── Serialisation ─────────────────────────────────────────────────────────────

class TestSerialization:
    def test_roundtrip_preserves_vector(self):
        original = np.random.rand(384).astype(np.float32)
        blob = embedding_to_bytes(original)
        recovered = bytes_to_embedding(blob)
        np.testing.assert_array_almost_equal(original, recovered)

    def test_blob_is_bytes(self):
        v = np.ones(384, dtype=np.float32)
        assert isinstance(embedding_to_bytes(v), bytes)


# ── CacheStore ────────────────────────────────────────────────────────────────

class TestCacheStore:
    def test_miss_on_empty_store(self, cache_store):
        emb = _fake_embed("What is Python?")
        result = cache_store.lookup(emb, threshold=0.92)
        assert result is None

    def test_write_then_exact_hit(self, cache_store):
        """Exact same prompt embedding should always hit."""
        prompt = "What is Python?"
        emb = _fake_embed(prompt)

        cache_store.write(prompt, emb, "Python is a programming language.", ttl=None)
        result = cache_store.lookup(emb, threshold=0.92)

        assert result is not None
        assert result.response == "Python is a programming language."
        assert result.similarity == pytest.approx(1.0, abs=1e-4)

    def test_dissimilar_prompt_misses(self, cache_store):
        """A semantically unrelated prompt must not hit the cache."""
        prompt_a = "What is Python?"
        emb_a = _fake_embed(prompt_a)
        cache_store.write(prompt_a, emb_a, "Python is a language.", ttl=None)

        # Chocolate cake is completely unrelated → orthogonal embedding
        emb_b = _fake_embed("What is the best chocolate cake recipe?") * -1  # force dissimilar
        emb_b = emb_b / np.linalg.norm(emb_b)

        result = cache_store.lookup(emb_b, threshold=0.92)
        assert result is None

    def test_ttl_expiry(self, cache_store):
        """An entry with expired TTL must not be returned."""
        prompt = "Test TTL"
        emb = _fake_embed(prompt)
        # Write with TTL of 1 second
        cache_store.write(prompt, emb, "cached response", ttl=1)

        # Should hit immediately
        assert cache_store.lookup(emb, threshold=0.90) is not None

        # Wait for expiry
        time.sleep(1.1)
        assert cache_store.lookup(emb, threshold=0.90) is None

    def test_hit_count_increments(self, cache_store):
        """hit_count must increment on each cache hit."""
        emb = _fake_embed("Python question")
        cache_store.write("Python question", emb, "answer", ttl=None)

        result1 = cache_store.lookup(emb, threshold=0.90)
        result2 = cache_store.lookup(emb, threshold=0.90)

        assert result1.hit_count == 1
        assert result2.hit_count == 2

    def test_count_returns_valid_entries(self, cache_store):
        """count() should return the number of non-expired entries."""
        assert cache_store.count() == 0
        cache_store.write("q1", _fake_embed("q1"), "a1", ttl=None)
        cache_store.write("q2", _fake_embed("q2"), "a2", ttl=None)
        assert cache_store.count() == 2

    def test_expired_entries_excluded_from_count(self, cache_store):
        """count() must not include expired entries."""
        cache_store.write("q", _fake_embed("q"), "a", ttl=1)
        assert cache_store.count() == 1
        time.sleep(1.1)
        assert cache_store.count() == 0
