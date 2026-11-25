from __future__ import annotations

import pytest

from ragkit.chunking import Chunk

pytest.importorskip("rank_bm25")

from ragkit.bm25 import BM25Index, tokenize  # noqa: E402


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id, "doc", "manual.md", 1, text, index=0, n_chars=len(text), n_words=len(text.split()))


CHUNKS = [
    _chunk("c1", "Alarm code E-41 indicates condenser water flow loss. Check the pump strainer."),
    _chunk("c2", "The CX-900 uses R-513A refrigerant with a charge of 480 kg per circuit."),
    _chunk("c3", "Consolidated revenue for fiscal year 2024 was $4,812 million."),
]


def test_tokenizer_keeps_codes_and_numbers():
    tokens = tokenize("Model CX-900 E-41 $4,812")
    assert "cx-900" in tokens and "cx900" in tokens
    assert "e-41" in tokens
    assert "4,812" in tokens


def test_tokenizer_drops_stopwords():
    assert "the" not in tokenize("the chiller is running")


def test_search_finds_exact_identifier():
    index = BM25Index.from_chunks(CHUNKS)
    hits = index.search("E-41", k=3)
    assert hits and hits[0].chunk_id == "c1"


def test_search_prefers_numeric_match():
    index = BM25Index.from_chunks(CHUNKS)
    hits = index.search("4,812 revenue", k=3)
    assert hits[0].chunk_id == "c3"


def test_empty_index_returns_nothing():
    assert BM25Index().search("anything") == []


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "bm25.pkl"
    index = BM25Index.from_chunks(CHUNKS, path)
    loaded = BM25Index.load(path)
    assert loaded is not None
    assert loaded.size == 3
    assert loaded.search("R-513A", k=1)[0].chunk_id == "c2"


def test_stale_index_is_rejected(tmp_path):
    path = tmp_path / "bm25.pkl"
    BM25Index.from_chunks(CHUNKS, path)
    assert BM25Index.load(path, expected_hash="different") is None
