"""Hybrid retrieval: dense (Chroma/Pinecone) + sparse (BM25) fused with RRF,
then re-ranked by a cross-encoder.

Why RRF and not score averaging? Dense cosine similarities (~0.6-0.9) and BM25
scores (0-30) live on incomparable scales. Reciprocal Rank Fusion only looks at
*rank*, so it needs no calibration and no per-corpus tuning - a property you want
in production.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from .bm25 import BM25Index
from .config import Settings
from .logging_utils import get_logger
from .rerank import Reranker
from .vectorstore import VectorStore

log = get_logger("hybrid")

RRF_K = 60


@dataclass
class Retrieved:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float = 0.0
    dense_score: float = 0.0
    bm25_score: float = 0.0
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["snippet"] = " ".join(self.text.split())[:300]
        # Keep a bounded excerpt: the evaluation harness needs the full evidence
        # to compute hit-rate/MRR, but reports should not explode in size.
        data["text"] = self.text[:1500]
        return data

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))

    @property
    def page(self) -> int:
        try:
            return int(self.metadata.get("page", 1))
        except (TypeError, ValueError):
            return 1


@dataclass
class HybridResult:
    query: str
    retrieved: list[Retrieved] = field(default_factory=list)
    kept: list[Retrieved] = field(default_factory=list)
    reranker_fallback: bool = False

    @property
    def top_score(self) -> float:
        return max((r.score for r in self.retrieved), default=0.0)


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        bm25: BM25Index,
        embedder: Any,
        reranker: Reranker,
        settings: Settings,
        chunk_lookup: dict[str, Any] | None = None,
    ) -> None:
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings
        # BM25 stores text only; this restores source/page metadata for
        # sparse-only hits so citations never say "unknown".
        self.chunk_lookup = chunk_lookup or {}

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: str,
        k: int | None = None,
        source: str | None = None,
        apply_threshold: bool = True,
    ) -> HybridResult:
        settings = self.settings
        k = k or settings.top_k_final
        query = (query or "").strip()
        if not query:
            return HybridResult(query=query)

        dense_hits = []
        try:
            vector = self.embedder.encode_one(query, is_query=True)
            dense_hits = self.store.search(vector, settings.top_k_dense, source=source)
        except Exception as exc:
            log.warning("dense retrieval failed (%s); continuing with BM25 only", type(exc).__name__)

        sparse_hits = self.bm25.search(query, settings.top_k_bm25) if self.bm25.is_ready else []

        # ---- Reciprocal Rank Fusion ----
        fused: dict[str, Retrieved] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            fused[hit.chunk_id] = Retrieved(
                chunk_id=hit.chunk_id,
                text=hit.text,
                metadata=dict(hit.metadata),
                dense_score=round(hit.score, 4),
                dense_rank=rank,
                rrf_score=1.0 / (RRF_K + rank),
            )
        for rank, hit in enumerate(sparse_hits, start=1):
            entry = fused.get(hit.chunk_id)
            contribution = 1.0 / (RRF_K + rank)
            if entry is None:
                chunk = self.chunk_lookup.get(hit.chunk_id)
                fused[hit.chunk_id] = Retrieved(
                    chunk_id=hit.chunk_id,
                    text=chunk.text if chunk else self.bm25.text_for(hit.chunk_id),
                    metadata=chunk.metadata() if chunk else {},
                    bm25_score=round(hit.score, 4),
                    bm25_rank=rank,
                    rrf_score=contribution,
                )
            else:
                entry.bm25_score = round(hit.score, 4)
                entry.bm25_rank = rank
                entry.rrf_score += contribution

        candidates = sorted(fused.values(), key=lambda r: r.rrf_score, reverse=True)
        pool = candidates[: max(settings.rerank_pool, k)]

        # ---- cross-encoder re-rank ----
        if pool:
            scores = self.reranker.score(query, [c.text for c in pool])
            for candidate, score in zip(pool, scores):
                candidate.score = float(score)
            pool.sort(key=lambda r: (r.score, r.rrf_score), reverse=True)

        kept = [c for c in pool if c.score >= settings.rerank_min_score] if apply_threshold else pool
        kept = kept[:k]

        return HybridResult(
            query=query,
            retrieved=pool,
            kept=kept,
            reranker_fallback=self.reranker.using_fallback,
        )

    def build_context(self, retrieved: Sequence[Retrieved], max_chars: int = 7000) -> str:
        """Numbered context blocks. Numbers match the citation markers the model emits."""
        blocks: list[str] = []
        used = 0
        for i, item in enumerate(retrieved, start=1):
            label = f"[{i}] {item.source} (page {item.page})"
            if item.metadata.get("heading"):
                label += f" — {item.metadata['heading']}"
            block = f"{label}\n{item.text.strip()}"
            if used + len(block) > max_chars and blocks:
                break
            blocks.append(block)
            used += len(block) + 2
        return "\n\n".join(blocks)
