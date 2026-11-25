"""Sentence-embedding wrapper around sentence-transformers.

`BAAI/bge-small-en-v1.5` is 33M parameters / 384 dims: it runs comfortably on a
6 GB RTX 2060 (or CPU-only Codespaces) and still ranks well on financial text.
"""

from __future__ import annotations

import threading
import zlib
from typing import Sequence

import numpy as np

from .logging_utils import get_logger

log = get_logger("embeddings")

# bge v1.5 retrieval instruction (query side only).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingError(RuntimeError):
    pass


class Embedder:
    """Thread-safe, lazily loaded sentence embedder with L2-normalised output."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        query_prefix: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.query_prefix = BGE_QUERY_PREFIX if query_prefix is None else query_prefix
        self._model = None
        self._lock = threading.Lock()
        self._dim: int | None = None

    # ------------------------------------------------------------------ model
    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # pragma: no cover - heavy optional dep
                raise EmbeddingError(
                    "sentence-transformers is not installed. Run: pip install -r requirements.txt"
                ) from exc
            log.info("loading embedding model %s (first run downloads ~130 MB)", self.model_name)
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:
                raise EmbeddingError(
                    f"Could not load embedding model '{self.model_name}'. "
                    "Check network access to huggingface.co or set HF_HOME to a warm cache."
                ) from exc
            self._dim = int(self._model.get_sentence_embedding_dimension() or 0)
            log.info("embedding model ready (dim=%s)", self._dim)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._ensure_model()
        return int(self._dim or 0)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def warm(self) -> None:
        self._ensure_model()

    # --------------------------------------------------------------- encoding
    def encode(
        self,
        texts: Sequence[str],
        is_query: bool = False,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        texts = [t if isinstance(t, str) else str(t) for t in texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefix = self.query_prefix if (is_query and self.query_prefix) else ""
        prepared = [prefix + t for t in texts] if prefix else texts
        model = self._ensure_model()
        vectors = model.encode(
            prepared,
            batch_size=batch_size or self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_one(self, text: str, is_query: bool = False) -> np.ndarray:
        return self.encode([text], is_query=is_query)[0]


class HashEmbedder:
    """Deterministic offline embedder - a *test double*, never used in the app.

    It hashes character n-grams into a fixed space so retrieval tests can run
    with no network and no 130 MB download, while still behaving like a
    similarity model (shared vocabulary -> higher cosine).
    """

    model_name = "hash-embedder-test-double"
    query_prefix = ""
    is_loaded = True

    def __init__(self, dim: int = 512) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def warm(self) -> None:
        return None

    def _vec(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
        for token in tokens:
            for n in (3, 4):
                for i in range(max(1, len(token) - n + 1)):
                    gram = token[i : i + n]
                    h = zlib.crc32(f"{n}:{gram}".encode("utf-8")) % self._dim
                    vec[h] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def encode(self, texts: Sequence[str], is_query: bool = False, **_: object) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts])

    def encode_one(self, text: str, is_query: bool = False) -> np.ndarray:
        return self._vec(text)
