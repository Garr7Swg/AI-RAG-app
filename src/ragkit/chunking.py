"""Semantic chunking.

Why not fixed 1000-character windows? Financial tables and engineering
procedure steps lose their anchors. This chunker:

1. splits on markdown headings (structure aware - a 10-K section stays together),
2. splits each section into sentences,
3. embeds the sentences and computes cosine distance between neighbours,
4. breaks where the distance is in the top ``100 - percentile`` (semantic drift),
5. then enforces a hard character ceiling and a minimum chunk size.

The resulting chunks are semantically coherent *and* bounded, which is what
makes both dense retrieval and BM25 behave.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .loaders import Page
from .logging_utils import get_logger
from .security import neutralize_chat_tokens, scan_for_injection

log = get_logger("chunking")

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|(?<=[.!?])\n+")
_HEADING_RE = re.compile(r"(?m)^(#{1,4})\s+(.+?)\s*$")
_PLAIN_HEADING_RE = re.compile(
    r"(?m)^(?P<line>(?:Section\s+\d+(?:\.\d+)*|ARTICLE\s+[IVXLC]+|COMPLIANCE\s+CERTIFICATE|APPENDIX\s+[A-Z])\.?[^\n]{0,120})$"
)
_NUMERIC_ROW_RE = re.compile(r"(?m)^\s*[\w\s()&/.-]{2,60}\|\s*[-$(\d]")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    page: int
    text: str
    heading: str = ""
    index: int = 0
    n_chars: int = 0
    n_words: int = 0
    injection_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            source=data["source"],
            page=int(data.get("page", 1)),
            text=data["text"],
            heading=data.get("heading", ""),
            index=int(data.get("index", 0)),
            n_chars=int(data.get("n_chars", len(data.get("text", "")))),
            n_words=int(data.get("n_words", len(data.get("text", "").split()))),
            injection_flags=list(data.get("injection_flags") or []),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "page": self.page,
            "heading": self.heading[:200],
            "index": self.index,
            "chunk_id": self.chunk_id,
            "injection_flags": ",".join(self.injection_flags),
        }


def make_chunk_id(doc_id: str, page: int, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{doc_id}:{page}:{index}:{text[:200]}".encode("utf-8")).hexdigest()
    return f"{doc_id}-{page:04d}-{index:03d}-{digest[:8]}"


def split_sentences(text: str) -> list[str]:
    """Sentence splitter that keeps table rows and short procedure steps intact."""
    text = (text or "").strip()
    if not text:
        return []
    lines = [ln.strip() for ln in text.split("\n")]
    units: list[str] = []
    buffer: list[str] = []
    for line in lines:
        if not line:
            if buffer:
                units.append(" ".join(buffer))
                buffer = []
            continue
        is_structural = bool(
            _NUMERIC_ROW_RE.match(line)
            or line.startswith(("-", "*", "•"))
            or line.startswith(("Step ", "STEP ", "Table "))
            or len(line) < 60
        )
        sentences = [s.strip() for s in _SENTENCE_RE.split(line) if s.strip()]
        if is_structural or len(sentences) <= 1:
            if buffer:
                units.append(" ".join(buffer))
                buffer = []
            units.append(line)
        else:
            buffer.extend(sentences)
    if buffer:
        units.append(" ".join(buffer))
    return [u for u in units if u.strip()]


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split text into ``(heading, body)`` sections.

    Markdown headings are used when present. Plain-text contracts and manuals
    (``Section 7.1``, ``ARTICLE VIII``, ``COMPLIANCE CERTIFICATE``) get the same
    treatment - otherwise an entire credit agreement collapses into one chunk and
    the compliance certificate is buried at the end of an unrelated section.
    """
    matches = list(_HEADING_RE.finditer(text or ""))
    if not matches:
        return _split_plain_sections(text or "")
    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))
    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections.append((heading, body))
    return sections


def _split_plain_sections(text: str) -> list[tuple[str, str]]:
    matches = list(_PLAIN_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]
    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))
    for i, match in enumerate(matches):
        line = match.group("line").strip()
        # "Section 7.1 Maximum Leverage. The Borrower shall not ..." carries body
        # text on the heading line; keep the sentence in the body.
        heading, _, remainder = line.partition(". ")
        if not remainder or len(heading) > 90:
            heading, remainder = line[:90], ""
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = f"{remainder}\n{text[body_start:body_end]}".strip()
        sections.append((heading.strip(), body))
    return sections


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


def _hard_split(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    sentences = split_sentences(text) or [text]
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current}\n{sentence}".strip()
            continue
        if current:
            pieces.append(current)
        if len(sentence) > max_chars:
            step = max_chars
            for i in range(0, len(sentence), step):
                pieces.append(sentence[i : i + step].strip())
            current = ""
        else:
            current = sentence
    if current:
        pieces.append(current)
    return [p for p in pieces if p.strip()]


def _semantic_units(units: Sequence[str], embedder: Any, percentile: float, max_chars: int) -> list[list[str]]:
    """Group sentences into semantically coherent groups."""
    if len(units) <= 1:
        return [list(units)]
    try:
        vectors = embedder.encode(list(units))
    except Exception as exc:  # model failure must not break ingestion
        log.warning("semantic grouping fell back to size-based split: %s", type(exc).__name__)
        return _size_groups(units, max_chars)

    distances = np.array(
        [_cosine_distance(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)],
        dtype=np.float32,
    )
    if distances.size == 0 or float(distances.max() - distances.min()) < 1e-6:
        return _size_groups(units, max_chars)

    threshold = float(np.percentile(distances, percentile))
    groups: list[list[str]] = []
    current: list[str] = [units[0]]
    for i, unit in enumerate(units[1:], start=1):
        boundary = float(distances[i - 1]) >= threshold
        current_len = sum(len(u) + 1 for u in current)
        overflow = current_len + len(unit) + 1 > max_chars
        if (boundary or overflow) and current:
            groups.append(current)
            current = [unit]
        else:
            current.append(unit)
    if current:
        groups.append(current)
    return groups


def _size_groups(units: Sequence[str], max_chars: int) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    length = 0
    for unit in units:
        if current and length + len(unit) + 1 > max_chars:
            groups.append(current)
            current, length = [], 0
        current.append(unit)
        length += len(unit) + 1
    if current:
        groups.append(current)
    return groups


def chunk_pages(
    pages: Iterable[Page],
    embedder: Any | None = None,
    max_chars: int = 1400,
    min_chars: int = 240,
    percentile: float = 88.0,
) -> list[Chunk]:
    """Turn pages into :class:`Chunk` objects.

    ``embedder`` may be ``None`` (pure size-based fallback) or any object with an
    ``encode(list[str]) -> np.ndarray`` method - which is how tests stay offline.
    """
    chunks: list[Chunk] = []
    for page in pages:
        text = neutralize_chat_tokens(page.text)
        findings = scan_for_injection(text)
        flags = sorted({f.kind for f in findings})
        index = 0
        for heading, body in split_sections(text):
            if not body.strip():
                continue
            units = split_sentences(body)
            if not units:
                continue
            groups = _semantic_units(units, embedder, percentile, max_chars) if embedder else _size_groups(units, max_chars)

            # merge a tiny trailing group into its predecessor
            if len(groups) > 1 and sum(len(u) for u in groups[-1]) < min_chars // 2:
                groups[-2].extend(groups[-1])
                groups.pop()

            for group in groups:
                # Join with newlines, not spaces: table rows and numbered steps must
                # survive into the prompt and into citations. Flattening them turns a
                # financial table into an unreadable run-on sentence.
                body_text = "\n".join(group).strip()
                if not body_text:
                    continue
                for piece in _hard_split(body_text, max_chars):
                    if not piece.strip():
                        continue
                    content = f"{heading}\n{piece}".strip() if heading else piece
                    if len(content) < 20:
                        continue
                    chunks.append(
                        Chunk(
                            chunk_id=make_chunk_id(page.doc_id, page.page, index, content),
                            doc_id=page.doc_id,
                            source=page.source,
                            page=page.page,
                            text=content,
                            heading=heading,
                            index=index,
                            n_chars=len(content),
                            n_words=len(content.split()),
                            injection_flags=flags,
                        )
                    )
                    index += 1
        if flags:
            log.warning("%s p.%s flagged for possible prompt injection: %s", page.source, page.page, flags)
    log.info("produced %d chunks", len(chunks))
    return chunks


def merge_small_chunks(chunks: Sequence[Chunk], min_chars: int = 240) -> list[Chunk]:
    """Post-pass: join orphaned micro-chunks with the following chunk of the same page."""
    merged: list[Chunk] = []
    for chunk in chunks:
        if (
            merged
            and chunk.n_chars < min_chars
            and merged[-1].doc_id == chunk.doc_id
            and merged[-1].page == chunk.page
            and merged[-1].n_chars + chunk.n_chars <= min_chars * 2
        ):
            previous = merged.pop()
            combined = f"{previous.text}\n{chunk.text}"
            merged.append(
                Chunk(
                    chunk_id=previous.chunk_id,
                    doc_id=previous.doc_id,
                    source=previous.source,
                    page=previous.page,
                    text=combined,
                    heading=previous.heading,
                    index=previous.index,
                    n_chars=len(combined),
                    n_words=len(combined.split()),
                    injection_flags=sorted(set(previous.injection_flags) | set(chunk.injection_flags)),
                )
            )
        else:
            merged.append(chunk)
    return merged
