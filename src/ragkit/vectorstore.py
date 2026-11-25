"""Vector stores: ChromaDB (default, zero cost, local) and Pinecone (free tier).

Both implement the same tiny interface so the rest of the pipeline is agnostic.
Embeddings are always computed by us (never by the store), which keeps the
embedding model identical between ingestion and query time - a classic source of
silent retrieval regressions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .chunking import Chunk
from .config import Settings
from .logging_utils import get_logger

log = get_logger("vectorstore")


@dataclass
class Hit:
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int: ...
    def search(self, vector: np.ndarray, k: int, source: str | None = None) -> list[Hit]: ...
    def delete_doc(self, doc_id: str) -> None: ...
    def count(self) -> int: ...
    def reset(self) -> None: ...


# --------------------------------------------------------------------- shared
def _flatten(metadatas: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for meta in metadatas or []:
        clean: dict[str, Any] = {}
        for key, value in (meta or {}).items():
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif value is None:
                clean[key] = ""
            else:
                clean[key] = str(value)
        out.append(clean)
    return out


def load_chunks(path: Path) -> list[Chunk]:
    path = Path(path)
    if not path.exists():
        return []
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(Chunk.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    return chunks


def save_chunks(chunks: Sequence[Chunk], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    tmp.replace(path)


# --------------------------------------------------------------------- chroma
class ChromaStore:
    def __init__(self, directory: Path, collection_name: str) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.directory),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self.name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "description": "enterprise RAG chunks"},
        )
        log.info("Chroma collection '%s' ready (%d vectors)", collection_name, self.count())

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        if len(chunks) == 0:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError("chunks/vectors length mismatch")
        batch = 256
        written = 0
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            self._collection.upsert(
                ids=[c.chunk_id for c in part],
                embeddings=vectors[start : start + batch].astype(np.float32).tolist(),
                documents=[c.text for c in part],
                metadatas=_flatten([c.metadata() for c in part]),
            )
            written += len(part)
        return written

    def search(self, vector: np.ndarray, k: int, source: str | None = None) -> list[Hit]:
        where = {"source": source} if source else None
        result = self._collection.query(
            query_embeddings=[vector.astype(np.float32).tolist()],
            n_results=max(1, k),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        hits: list[Hit] = []
        for i, chunk_id in enumerate(ids):
            distance = float(dists[i]) if i < len(dists) else 1.0
            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    score=max(0.0, 1.0 - distance),  # cosine distance -> similarity
                    text=docs[i] if i < len(docs) else "",
                    metadata=_flatten([metas[i] if i < len(metas) else {}])[0],
                )
            )
        return hits

    def delete_doc(self, doc_id: str) -> None:
        try:
            self._collection.delete(where={"doc_id": doc_id})
        except Exception as exc:  # pragma: no cover
            log.warning("delete_doc(%s) failed: %s", doc_id, type(exc).__name__)

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception:
            return 0

    def reset(self) -> None:
        try:
            self._client.delete_collection(self.name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.name, metadata={"hnsw:space": "cosine"}
        )


# ------------------------------------------------------------------- pinecone
class PineconeStore:
    """Free-tier serverless Pinecone store (optional dependency)."""

    def __init__(self, api_key: str, index_name: str, dim: int, cloud: str, region: str) -> None:
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pinecone is not installed. Run: pip install pinecone") from exc
        self._pc = Pinecone(api_key=api_key)
        existing = {i["name"] for i in self._pc.list_indexes()}
        if index_name not in existing:
            log.info("creating Pinecone index '%s' (dim=%d)", index_name, dim)
            self._pc.create_index(
                name=index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        self._index = self._pc.Index(index_name)
        self.name = index_name

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        payload = [
            {
                "id": c.chunk_id,
                "values": vectors[i].astype(float).tolist(),
                "metadata": {**c.metadata(), "text": c.text[:35000]},
            }
            for i, c in enumerate(chunks)
        ]
        for start in range(0, len(payload), 100):
            self._index.upsert(vectors=payload[start : start + 100])
        return len(payload)

    def search(self, vector: np.ndarray, k: int, source: str | None = None) -> list[Hit]:
        result = self._index.query(
            vector=vector.astype(float).tolist(),
            top_k=max(1, k),
            include_metadata=True,
            filter={"source": {"$eq": source}} if source else None,
        )
        hits = []
        for match in result.get("matches", []):
            meta = dict(match.get("metadata") or {})
            text = meta.pop("text", "")
            hits.append(Hit(chunk_id=match["id"], score=float(match.get("score", 0.0)), text=text, metadata=meta))
        return hits

    def delete_doc(self, doc_id: str) -> None:
        try:
            self._index.delete(filter={"doc_id": {"$eq": doc_id}})
        except Exception as exc:  # pragma: no cover
            log.warning("pinecone delete_doc failed: %s", type(exc).__name__)

    def count(self) -> int:
        try:
            return int(self._index.describe_index_stats().get("total_vector_count", 0))
        except Exception:
            return 0

    def reset(self) -> None:
        try:
            self._index.delete(delete_all=True)
        except Exception:
            pass


def get_store(settings: Settings, dim: int) -> VectorStore:
    if settings.vector_backend == "pinecone":
        if not settings.pinecone_api_key:
            raise RuntimeError("RAG_VECTOR_BACKEND=pinecone requires PINECONE_API_KEY.")
        return PineconeStore(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index,
            dim=dim,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )
    return ChromaStore(settings.chroma_dir, settings.collection_name)
