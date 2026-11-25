"""BM25 keyword index.

Keyword search is not optional in enterprise RAG: exact identifiers
("covenant 7.2(b)", "E-27", "$4.812 billion") are where dense retrieval quietly
fails. BM25 catches them, RRF fuses the two rankings.
"""

from __future__ import annotations

import hashlib
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .chunking import Chunk
from .logging_utils import get_logger

log = get_logger("bm25")

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._%,/-]*")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were",
    "will", "with", "this", "these", "those", "we", "our", "us", "which", "what",
}
_INDEX_VERSION = 2


def tokenize(text: str) -> list[str]:
    """Lowercase tokenizer that keeps numbers, codes and hyphenated ids intact."""
    tokens = []
    for raw in _TOKEN_RE.findall((text or "").lower()):
        token = raw.strip("._-/,")
        if not token:
            continue
        if token in _STOPWORDS:
            continue
        tokens.append(token)
        # "$4,812" also matches a chunk written as "4812"
        if "," in token:
            tokens.append(token.replace(",", ""))
        # split invoice-like ids so "cx-900" also matches "cx900"
        if "-" in token:
            joined = token.replace("-", "")
            if joined:
                tokens.append(joined)
    return tokens


@dataclass
class BM25Hit:
    chunk_id: str
    score: float
    rank: int


class BM25Index:
    """Thin wrapper over rank_bm25 with disk persistence."""

    def __init__(self) -> None:
        self._bm25 = None
        self._ids: list[str] = []
        self._texts: dict[str, str] = {}
        self._corpus_hash: str = ""

    # ------------------------------------------------------------------ build
    def build(self, chunks: Sequence[Chunk]) -> "BM25Index":
        from rank_bm25 import BM25Okapi

        self._ids = [c.chunk_id for c in chunks]
        self._texts = {c.chunk_id: c.text for c in chunks}
        corpus = [tokenize(c.text) or ["_empty_"] for c in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        digest = hashlib.sha256()
        for chunk in chunks:
            digest.update(chunk.chunk_id.encode())
        self._corpus_hash = digest.hexdigest()
        log.info("BM25 index built over %d chunks", len(self._ids))
        return self

    @property
    def size(self) -> int:
        return len(self._ids)

    @property
    def corpus_hash(self) -> str:
        return self._corpus_hash

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None and bool(self._ids)

    # ----------------------------------------------------------------- search
    def search(self, query: str, k: int = 20) -> list[BM25Hit]:
        if not self.is_ready:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)  # type: ignore[union-attr]
        order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[: max(1, k)]
        hits: list[BM25Hit] = []
        for rank, idx in enumerate(order, start=1):
            score = float(scores[idx])
            if score <= 0:
                break
            hits.append(BM25Hit(chunk_id=self._ids[idx], score=score, rank=rank))
        return hits

    def text_for(self, chunk_id: str) -> str:
        return self._texts.get(chunk_id, "")

    # -------------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _INDEX_VERSION,
            "corpus_hash": self._corpus_hash,
            "ids": self._ids,
            "texts": self._texts,
            "bm25": self._bm25,
        }
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path, expected_hash: str | None = None) -> "BM25Index | None":
        path = Path(path)
        if not path.exists():
            return None
        try:
            with path.open("rb") as fh:
                payload = pickle.load(fh)  # noqa: S301 - local artefact we wrote ourselves
        except Exception as exc:
            log.warning("could not load BM25 index (%s); rebuilding", type(exc).__name__)
            return None
        if payload.get("version") != _INDEX_VERSION:
            return None
        if expected_hash and payload.get("corpus_hash") != expected_hash:
            log.warning("BM25 index is stale; rebuilding")
            return None
        index = cls()
        index._ids = payload["ids"]
        index._texts = payload["texts"]
        index._bm25 = payload["bm25"]
        index._corpus_hash = payload.get("corpus_hash", "")
        return index

    @classmethod
    def from_chunks(cls, chunks: Iterable[Chunk], path: Path | None = None) -> "BM25Index":
        index = cls().build(list(chunks))
        if path is not None:
            index.save(path)
        return index
