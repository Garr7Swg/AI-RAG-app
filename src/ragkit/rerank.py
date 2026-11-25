"""Cross-encoder re-ranking (free, local, ~22M params).

Bi-encoders fetch candidates fast but judge relevance coarsely. A cross-encoder
scores (query, chunk) jointly, which is where the biggest quality jump comes from
in a hybrid pipeline. ``cross-encoder/ms-marco-MiniLM-L-6-v2`` adds ~5 ms/chunk on
CPU and is 90 MB, so it is free-tier friendly.
"""

from __future__ import annotations

import re
import threading
from typing import Sequence

from .logging_utils import get_logger

log = get_logger("rerank")

_WORD_RE = re.compile(r"[a-z0-9]+")


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = 2.718281828459045 ** (-x)
        return 1.0 / (1.0 + z)
    z = 2.718281828459045**x
    return z / (1.0 + z)


def lexical_overlap(query: str, text: str) -> float:
    """Fallback relevance score: IDF-free token coverage of the query."""
    q = {t for t in _WORD_RE.findall((query or "").lower()) if len(t) > 2}
    if not q:
        return 0.0
    d = set(_WORD_RE.findall((text or "").lower()))
    return len(q & d) / len(q)


class Reranker:
    """Lazy cross-encoder with a deterministic lexical fallback."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._lock = threading.Lock()
        self.using_fallback = False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def warm(self) -> None:
        self._ensure()

    def _ensure(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import CrossEncoder

                log.info("loading reranker %s", self.model_name)
                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as exc:
                self.using_fallback = True
                log.warning(
                    "reranker unavailable (%s); using lexical fallback scores. "
                    "Install sentence-transformers or warm the HF cache for full quality.",
                    type(exc).__name__,
                )
                self._model = False  # sentinel: tried and failed
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return relevance scores in [0, 1] for each passage."""
        if not passages:
            return []
        model = self._ensure()
        if not model:
            self.using_fallback = True
            return [round(lexical_overlap(query, p), 4) for p in passages]
        pairs = [(query, p) for p in passages]
        try:
            raw = model.predict(pairs, show_progress_bar=False, convert_to_numpy=True)
        except Exception as exc:  # pragma: no cover
            log.warning("rerank prediction failed (%s); falling back", type(exc).__name__)
            self.using_fallback = True
            return [round(lexical_overlap(query, p), 4) for p in passages]
        return [round(_sigmoid(float(value)), 4) for value in raw]
