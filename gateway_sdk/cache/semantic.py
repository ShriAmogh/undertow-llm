"""
gateway_sdk.cache.semantic
==========================
Embedding engine for semantic cache.

How it works:
    1. On first call, the model is loaded from disk (or downloaded ~80MB).
       We load it lazily — no startup cost if caching is disabled.
    2. `embed(text)` converts a string into a 384-dimensional float32 vector.
    3. `cosine_similarity(a, b)` measures the angle between two vectors.
       Score = 1.0 means identical meaning; 0.0 means completely unrelated.

Warm-up strategy:
    The model takes ~8-10 s to load on first use (weights parsed from disk).
    Calling `prewarm()` starts a daemon background thread that loads the model
    immediately, so it is ready by the time the first real request arrives.
    `configure(cache_enabled=True)` calls prewarm() automatically.

Why cosine similarity (not Euclidean distance)?
    Embedding vectors vary in magnitude based on sentence length, not meaning.
    Cosine similarity normalises for magnitude — only the *direction* (meaning)
    matters. "Python" and "Tell me about Python" point in nearly the same
    direction even though their raw vectors differ in length.

Model: all-MiniLM-L6-v2
    - 384 dimensions (compact, fast)
    - ~80MB on disk (fits in RAM easily)
    - Runs entirely locally — no API cost, no network latency
    - Inference: ~5ms per prompt on CPU, <1ms on GPU
"""

from __future__ import annotations

import io
import threading
import numpy as np
from typing import Optional

# Lazy module-level reference — model loaded on first use
_model = None
_MODEL_NAME = "all-MiniLM-L6-v2"

# Lock ensures only one thread loads the model even under concurrent embed() calls
_model_lock = threading.Lock()
# Reference to the active prewarm thread (if any) so _get_model() can join it
_prewarm_thread: threading.Thread | None = None


def _get_model():
    """Load the sentence-transformer model once and cache it in memory.

    Thread-safe: if two threads race to the first embed() call, the lock
    ensures the model is initialised exactly once (double-checked locking).
    If prewarm() already started a background load, we join that thread
    instead of starting a duplicate load.
    """
    global _model
    if _model is None:
        # If a prewarm thread is in progress and we are NOT that thread, wait
        # for it — it will set _model.  Guard against self-join (the prewarm
        # thread itself calls _get_model(), so it must skip the join).
        if (
            _prewarm_thread is not None
            and _prewarm_thread.is_alive()
            and threading.current_thread() is not _prewarm_thread
        ):
            _prewarm_thread.join()
        # If prewarm wasn't used (or race finished), load synchronously under lock
        if _model is None:
            with _model_lock:
                if _model is None:  # double-checked inside lock
                    from sentence_transformers import SentenceTransformer
                    _model = SentenceTransformer(_MODEL_NAME)
    return _model


def prewarm() -> None:
    """Pre-load the embedding model in a background daemon thread.

    Call this once at startup (configure() does it automatically when
    cache_enabled=True).  By the time the first real @track call arrives
    the model is already in memory and embed() returns in <5 ms instead
    of blocking for ~8-10 s.

    Safe to call multiple times — subsequent calls are no-ops if the model
    is already loaded or currently being loaded.
    """
    global _prewarm_thread

    if _model is not None:
        return  # already warm, nothing to do
    if _prewarm_thread is not None and _prewarm_thread.is_alive():
        return  # prewarm already in flight

    def _load() -> None:
        _get_model()  # blocks only this daemon thread; main thread continues

    _prewarm_thread = threading.Thread(
        target=_load,
        name="gateway-sdk-embed-prewarm",
        daemon=True,   # won't prevent the process from exiting
    )
    _prewarm_thread.start()


def embed(text: str) -> np.ndarray:
    """
    Convert a text string into a 384-dim float32 embedding vector.

    Args:
        text: The prompt or text to embed.

    Returns:
        A numpy array of shape (384,), dtype float32.
    """
    model = _get_model()
    # encode() returns a numpy array; normalize_embeddings=True makes cosine
    # similarity equivalent to a simple dot product (minor optimisation)
    vector = model.encode(text, normalize_embeddings=True)
    return vector.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Because we normalise embeddings in embed(), this is a simple dot product.
    Score range: [-1, 1], but practically [0, 1] for natural language.
      - 1.0 = identical meaning
      - 0.92+ = semantically very similar (good cache hit threshold)
      - < 0.7 = different topics

    Args:
        a: First embedding vector (shape: 384,)
        b: Second embedding vector (shape: 384,)

    Returns:
        Float in [-1, 1].
    """
    # np.dot on unit vectors = cosine similarity
    return float(np.dot(a, b))


# ── Serialisation helpers (for storing embeddings as SQLite BLOBs) ────────────

def embedding_to_bytes(vector: np.ndarray) -> bytes:
    """Serialise a numpy embedding to bytes for SQLite BLOB storage."""
    buf = io.BytesIO()
    np.save(buf, vector)
    return buf.getvalue()


def bytes_to_embedding(blob: bytes) -> np.ndarray:
    """Deserialise a SQLite BLOB back into a numpy embedding vector."""
    buf = io.BytesIO(blob)
    return np.load(buf)
