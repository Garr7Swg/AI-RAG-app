"""Ingestion: pages -> semantic chunks -> embeddings -> Chroma/Pinecone + BM25.

Idempotent by design: re-ingesting the same document replaces its vectors and
chunks (content-addressed ``doc_id``), so the demo can be rebuilt as often as you
like without duplicating evidence.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .chunking import Chunk, chunk_pages, merge_small_chunks
from .config import Settings, get_settings
from .embeddings import Embedder
from .loaders import Page
from .logging_utils import get_logger
from .vectorstore import VectorStore, get_store, load_chunks, save_chunks

log = get_logger("ingest")

MANIFEST_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_manifest(settings: Settings) -> dict[str, Any]:
    path = settings.manifest_path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


@dataclass
class IngestReport:
    documents: int = 0
    pages: int = 0
    chunks: int = 0
    seconds: float = 0.0
    sources: list[str] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)
    total_chunks: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "pages": self.pages,
            "chunks": self.chunks,
            "total_chunks": self.total_chunks,
            "seconds": round(self.seconds, 2),
            "sources": self.sources,
            "rejected": self.rejected,
        }


class Ingestor:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        progress: Callable[[str, float], None] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self._embedder = embedder
        self._store = store
        self.progress = progress

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.settings.embed_model)
        return self._embedder

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = get_store(self.settings, self.embedder.dim)
        return self._store

    def _notify(self, message: str, fraction: float) -> None:
        log.info("%s (%.0f%%)", message, fraction * 100)
        if self.progress:
            try:
                self.progress(message, max(0.0, min(1.0, fraction)))
            except Exception:  # pragma: no cover - UI callbacks must not break ingestion
                pass

    # ------------------------------------------------------------------ public
    def ingest_pages(self, pages: Sequence[Page], rebuild_bm25: bool = True) -> IngestReport:
        started = time.perf_counter()
        report = IngestReport(pages=len(pages))
        if not pages:
            report.seconds = time.perf_counter() - started
            self._notify("No readable pages found", 1.0)
            return report

        self._notify("Chunking documents", 0.1)
        chunks = chunk_pages(
            pages,
            embedder=self.embedder,
            max_chars=self.settings.chunk_max_chars,
            min_chars=self.settings.chunk_min_chars,
            percentile=self.settings.semantic_percentile,
        )
        chunks = merge_small_chunks(chunks, self.settings.chunk_min_chars)
        if not chunks:
            report.seconds = time.perf_counter() - started
            return report

        doc_ids = {c.doc_id for c in chunks}
        sources = sorted({c.source for c in chunks})
        report.documents = len(doc_ids)
        report.sources = sources
        pages_by_doc: dict[str, int] = {}
        for page in pages:
            if page.doc_id in doc_ids:
                pages_by_doc[page.doc_id] = pages_by_doc.get(page.doc_id, 0) + 1

        # 1. remove previous versions of these documents
        for doc_id in doc_ids:
            self.store.delete_doc(doc_id)
        existing = [c for c in load_chunks(self.settings.chunks_path) if c.doc_id not in doc_ids]

        # 2. embed in batches
        vectors = self._embed_chunks(chunks)

        # 3. write to the vector store
        self._notify("Writing vectors", 0.75)
        self.store.upsert(chunks, vectors)

        # 4. persist chunk text (source of truth for BM25 + citations)
        all_chunks = existing + list(chunks)
        save_chunks(all_chunks, self.settings.chunks_path)
        report.chunks = len(chunks)
        report.total_chunks = len(all_chunks)

        # 5. rebuild the keyword index
        if rebuild_bm25:
            self._notify("Rebuilding BM25 index", 0.9)
            from .bm25 import BM25Index

            BM25Index.from_chunks(all_chunks, self.settings.bm25_path)

        # 6. manifest
        manifest = load_manifest(self.settings)
        documents = dict(manifest.get("documents", {}))
        for chunk in chunks:
            entry = documents.get(chunk.doc_id, {})
            entry.update(
                {
                    "source": chunk.source,
                    "sha256": "",
                    "pages": pages_by_doc.get(chunk.doc_id, 0),
                    "chunks": sum(1 for c in all_chunks if c.doc_id == chunk.doc_id),
                    "ingested_at": _now(),
                    "injection_flags": sorted(set(entry.get("injection_flags", [])) | set(chunk.injection_flags)),
                }
            )
            documents[chunk.doc_id] = entry
        _write_json_atomic(
            self.settings.manifest_path,
            {
                "version": MANIFEST_VERSION,
                "built_at": _now(),
                "embed_model": self.settings.embed_model,
                "embed_dim": int(vectors.shape[1]) if vectors.size else self.embedder.dim,
                "chunk_max_chars": self.settings.chunk_max_chars,
                "chunk_count": len(all_chunks),
                "documents": documents,
            },
        )

        report.seconds = time.perf_counter() - started
        self._notify(f"Ingested {report.chunks} chunks", 1.0)
        return report

    def ingest_paths(self, paths: Iterable[Path]) -> IngestReport:
        from .loaders import load_path
        from .security import UploadRejected

        pages: list[Page] = []
        rejected: list[dict[str, str]] = []
        for path in paths:
            try:
                pages.extend(load_path(Path(path), self.settings.max_file_bytes))
            except UploadRejected as exc:
                rejected.append({"file": Path(path).name, "reason": exc.reason, "detail": exc.detail})
        report = self.ingest_pages(pages)
        report.rejected = rejected
        return report

    def delete_document(self, doc_id: str) -> int:
        self.store.delete_doc(doc_id)
        chunks = load_chunks(self.settings.chunks_path)
        kept = [c for c in chunks if c.doc_id != doc_id]
        removed = len(chunks) - len(kept)
        save_chunks(kept, self.settings.chunks_path)
        if kept:
            from .bm25 import BM25Index

            BM25Index.from_chunks(kept, self.settings.bm25_path)
        else:
            Path(self.settings.bm25_path).unlink(missing_ok=True)
        manifest = load_manifest(self.settings)
        documents = dict(manifest.get("documents", {}))
        documents.pop(doc_id, None)
        manifest.update({"documents": documents, "chunk_count": len(kept), "built_at": _now()})
        _write_json_atomic(self.settings.manifest_path, manifest)
        return removed

    def reset(self) -> None:
        """Delete every derived artefact (never the corpus)."""
        self.store.reset()
        for path in (self.settings.chunks_path, self.settings.bm25_path, self.settings.manifest_path):
            Path(path).unlink(missing_ok=True)
        log.info("index reset")

    # ----------------------------------------------------------------- private
    def _embed_chunks(self, chunks: Sequence[Chunk]) -> np.ndarray:
        batch = 64
        total = len(chunks)
        parts: list[np.ndarray] = []
        for start in range(0, total, batch):
            part = chunks[start : start + batch]
            self._notify(f"Embedding {start + len(part)}/{total} chunks", 0.15 + 0.55 * (start / max(1, total)))
            parts.append(self.embedder.encode([c.text for c in part]))
        vectors = np.vstack(parts) if parts else np.zeros((0, self.embedder.dim), dtype=np.float32)
        if len(vectors) != total:
            raise RuntimeError("Embedding count mismatch")
        return vectors.astype(np.float32)
