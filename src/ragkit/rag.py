"""The RAG pipeline: retrieve -> fence -> generate -> verify -> cite.

The verification stage is the part that makes this "enterprise": an answer is
only shown if it is (a) cited, (b) lexically grounded in the cited text, and
(c) does not contain numbers that are absent from the cited text. Otherwise the
pipeline returns exactly "I don't know." - which is a *feature*, not a failure.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Sequence

from .bm25 import BM25Index
from .config import Settings, get_settings
from .embeddings import Embedder
from .hybrid import HybridRetriever, HybridResult, Retrieved
from .ingest import Ingestor, load_manifest
from .llm import BaseLLM, LLMError, Message, get_llm
from .loaders import Page, iter_corpus
from .logging_utils import get_logger
from .prompts import IDK_TEXT, REFUSAL_MARKERS, SYSTEM_PROMPT, build_rewrite_prompt, build_user_prompt
from .rerank import Reranker
from .security import clean_question, neutralize_chat_tokens, snippet
from .vectorstore import get_store, load_chunks

log = get_logger("rag")

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’.-]*")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was",
    "were", "be", "been", "by", "with", "as", "at", "from", "that", "this", "these",
    "those", "it", "its", "which", "who", "whom", "what", "when", "where", "how",
    "not", "no", "if", "than", "then", "there", "their", "they", "we", "our", "you",
    "your", "i", "he", "she", "his", "her", "has", "have", "had", "do", "does", "did",
    "can", "could", "would", "should", "may", "might", "will", "about", "into", "per",
    "also", "any", "all", "such", "but", "so", "up", "out", "over", "under",
}


@dataclass
class Citation:
    index: int
    chunk_id: str
    source: str
    page: int
    snippet: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RAGAnswer:
    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    abstained: bool = False
    abstain_reason: str = ""
    latency_ms: float = 0.0
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    reranker_fallback: bool = False
    model: str = ""
    mode: str = "llm"

    def sources(self) -> list[dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        for citation in self.citations:
            seen.setdefault(
                f"{citation.source}:{citation.page}",
                {"source": citation.source, "page": citation.page, "score": citation.score},
            )
        return list(seen.values())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["citations"] = [c.to_dict() for c in self.citations]
        return data


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in (w.lower() for w in _WORD_RE.findall(text or ""))
        if token not in _STOPWORDS and len(token) > 1
    }


def numbers_in(text: str) -> list[str]:
    return [n.replace(" ", "") for n in _NUMBER_RE.findall(text or "")]


def looks_like_refusal(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def _stem(token: str) -> str:
    """Tiny suffix stripper so 'changed' matches 'change' in the extractive baseline."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > 5 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    return token.rstrip("e")


def _candidate_parts(text: str) -> list[str]:
    """Split chunk text into quotable units: lines, table rows and long sentences.

    Chunking joins table rows with ' | ', so a table arrives as one long line.
    Splitting on the row boundary ('| |') recovers readable, scoreable rows.
    """
    parts: list[str] = []
    for line in (text or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        if "|" in line:
            for row in re.split(r"\s*\|\s*\|\s*", line):
                cells = [cell.strip(" |-*") for cell in row.split("|")]
                cells = [cell for cell in cells if cell]
                if cells:
                    parts.append(" ".join(cells))
        elif len(line) > 320:
            parts.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+", line) if part.strip())
        else:
            parts.append(line)
    return parts


class RAGPipeline:
    """Owns retrieval + generation. Construct once, reuse across requests."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: BaseLLM | None = None,
        embedder: Embedder | None = None,
        store: Any | None = None,
        bm25: BM25Index | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._llm = llm
        self._embedder = embedder
        self._reranker = reranker
        self._store = store
        self._bm25 = bm25
        self._retriever: HybridRetriever | None = None
        self._load_error: str = ""

    # ------------------------------------------------------------------ wiring
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.settings.embed_model)
        return self._embedder

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = get_llm(self.settings)
        return self._llm

    @property
    def store(self):
        if self._store is None:
            self._store = get_store(self.settings, self.embedder.dim)
        return self._store

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            index = BM25Index.load(self.settings.bm25_path)
            if index is None:
                chunks = load_chunks(self.settings.chunks_path)
                if chunks:
                    index = BM25Index.from_chunks(chunks, self.settings.bm25_path)
                else:
                    index = BM25Index()
            self._bm25 = index
        return self._bm25

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            if self._reranker is None:
                self._reranker = Reranker(self.settings.rerank_model)
            lookup = {chunk.chunk_id: chunk for chunk in load_chunks(self.settings.chunks_path)}
            self._retriever = HybridRetriever(
                self.store,
                self.bm25,
                self.embedder,
                self._reranker,
                self.settings,
                chunk_lookup=lookup,
            )
        return self._retriever

    def warm(self) -> None:
        """Load models eagerly so the first question is not a cold start."""
        self.embedder.warm()
        if self._reranker is not None:
            self._reranker.warm()

    def health(self) -> dict[str, Any]:
        manifest = load_manifest(self.settings)
        ok, message = False, "Index not built. Run: make ingest"
        if manifest and manifest.get("chunk_count"):
            ok, message = True, f"{manifest['chunk_count']} chunks from {len(manifest.get('documents', {}))} documents"
        llm_ok, llm_message = (False, "LLM not checked")
        try:
            llm_ok, llm_message = self.llm.health()
        except Exception as exc:  # pragma: no cover
            llm_message = f"LLM check failed ({type(exc).__name__})"
        return {
            "index_ready": ok,
            "index_message": message,
            "llm_ready": llm_ok,
            "llm_message": llm_message,
            "embed_model": self.settings.embed_model,
            "llm_provider": self.settings.llm_provider,
            "documents": len(manifest.get("documents", {})) if manifest else 0,
            "chunks": manifest.get("chunk_count", 0) if manifest else 0,
        }

    # ------------------------------------------------------------------ ingest
    def ingestor(self) -> Ingestor:
        return Ingestor(self.settings, embedder=self.embedder, store=self.store)

    def ingest_corpus(self, rebuild: bool = False) -> dict[str, Any]:
        ingestor = self.ingestor()
        if rebuild:
            ingestor.reset()
        pages = iter_corpus(self.settings.corpus_dir, self.settings.max_file_bytes)
        report = ingestor.ingest_pages(pages)
        self._bm25 = None
        self._retriever = None
        return report.to_dict()

    def ingest_uploads(self, files: Sequence[tuple[str, bytes]]) -> dict[str, Any]:
        from .loaders import load_bytes
        from .security import UploadRejected

        pages: list[Page] = []
        rejected: list[dict[str, str]] = []
        for name, data in files:
            try:
                pages.extend(load_bytes(name, data, self.settings.max_file_bytes))
            except UploadRejected as exc:
                rejected.append({"file": name, "reason": exc.reason, "detail": exc.detail})
        report = self.ingestor().ingest_pages(pages)
        report.rejected = rejected
        self._bm25 = None
        self._retriever = None
        return report.to_dict()

    # ------------------------------------------------------------------ query
    def _standalone_question(self, question: str, history: Sequence[dict[str, str]] | None) -> str:
        if not history:
            return question
        recent = [h for h in history[-4:] if h.get("content")]
        if not recent:
            return question
        conversation = "\n".join(f"{h.get('role', 'user')}: {snippet(h['content'], 400)}" for h in recent)
        try:
            rewritten = self.llm.complete(
                [Message("system", "Rewrite follow-up questions as standalone search queries."),
                 Message("user", build_rewrite_prompt(question, conversation))],
                temperature=0.0,
            )
            cleaned = neutralize_chat_tokens(rewritten).strip().splitlines()[0]
            if 0 < len(cleaned) <= self.settings.max_question_chars:
                return cleaned
        except LLMError:
            pass
        return question

    def _retrieve(self, question: str) -> HybridResult:
        return self.retriever.search(question, k=self.settings.top_k_final)

    def _unmatched_identifiers(self, question: str, kept: Sequence[Retrieved]) -> list[str]:
        """Entity guard: a model number or clause id in the question must appear in
        the retrieved evidence. 'CX-1200' against a CX-900 manual must not be
        answered from nearby text."""
        codes = {
            token
            for token in content_tokens(question)
            if re.search(r"[a-z]{2,}[-_]?\d", token) or re.search(r"\d+\.\d+", token)
        }
        if not codes:
            return []
        haystack = " ".join(item.text.lower() for item in kept)
        return sorted(code for code in codes if code not in haystack)

    def _citations_from(self, indices: Sequence[int], kept: Sequence[Retrieved]) -> list[Citation]:
        citations: list[Citation] = []
        for number in indices:
            if 1 <= number <= len(kept):
                item = kept[number - 1]
                citations.append(
                    Citation(
                        index=number,
                        chunk_id=item.chunk_id,
                        source=item.source,
                        page=item.page,
                        snippet=snippet(item.text, 280),
                        score=round(item.score, 4),
                    )
                )
        # de-duplicate while preserving order
        unique: dict[int, Citation] = {}
        for citation in citations:
            unique.setdefault(citation.index, citation)
        return list(unique.values())

    def _verify(
        self,
        answer_text: str,
        kept: Sequence[Retrieved],
        citations: list[Citation],
        question: str = "",
    ) -> tuple[bool, str, list[Citation]]:
        """Return (ok, reason, citations). Any failure means abstain."""
        if not answer_text.strip():
            return False, "empty_answer", citations
        if looks_like_refusal(answer_text):
            return True, "model_abstained", []

        if not citations:
            # Try to attach the chunks that actually support the answer.
            answer_tokens = content_tokens(answer_text)
            scored: list[tuple[float, int]] = []
            for i, item in enumerate(kept, start=1):
                overlap = len(answer_tokens & content_tokens(item.text)) / max(1, len(answer_tokens))
                if overlap >= 0.5:
                    scored.append((overlap, i))
            scored.sort(reverse=True)
            citations = self._citations_from([i for _, i in scored[:2]], kept)
            if not citations:
                return False, "no_citations", []

        evidence = "\n".join(kept[c.index - 1].text for c in citations if 1 <= c.index <= len(kept))
        if not evidence:
            return False, "citation_out_of_range", []

        # Citation markers are not content: strip them before grounding/number checks.
        body = _CITATION_RE.sub(" ", answer_text)
        answer_tokens = content_tokens(body)
        if answer_tokens:
            covered = len(answer_tokens & content_tokens(evidence)) / len(answer_tokens)
            if covered < self.settings.grounding_min_overlap:
                return False, f"low_grounding:{covered:.2f}", citations

        if self.settings.strict_numeric_grounding:
            evidence_numbers = {n.rstrip(".,") for n in numbers_in(evidence)}
            # Numbers the user supplied ("fiscal 2024") are not hallucinations;
            # the model is allowed to restate them in the answer.
            question_numbers = {n.rstrip(".,") for n in numbers_in(question)}
            for number in numbers_in(body):
                if number and number not in evidence_numbers and number not in question_numbers:
                    return False, f"unsupported_number:{number}", citations

        return True, "ok", citations

    def answer(
        self,
        question: str,
        history: Sequence[dict[str, str]] | None = None,
    ) -> RAGAnswer:
        started = time.perf_counter()
        question = clean_question(question, self.settings.max_question_chars)
        search_query = self._standalone_question(question, history)
        result = self._retrieve(search_query)

        if not result.kept:
            return RAGAnswer(
                question=question,
                text=IDK_TEXT,
                abstained=True,
                abstain_reason="no_relevant_context",
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                retrieved=[r.to_dict() for r in result.retrieved[:8]],
                reranker_fallback=result.reranker_fallback,
                model=self.settings.llm_provider,
            )

        missing = self._unmatched_identifiers(question, result.kept)
        if missing:
            return RAGAnswer(
                question=question,
                text=IDK_TEXT,
                abstained=True,
                abstain_reason=f"unmatched_identifier:{','.join(missing)}",
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                retrieved=[r.to_dict() for r in result.retrieved[:8]],
                model=self.settings.llm_provider,
            )

        context = self.retriever.build_context(result.kept)
        has_injection = any("injection" in " ".join(r.metadata.get("injection_flags", "")) for r in result.kept)
        messages = [
            Message("system", SYSTEM_PROMPT),
            Message("user", build_user_prompt(question, context, has_injection)),
        ]
        try:
            raw = self.llm.complete(messages, temperature=self.settings.temperature)
        except LLMError as exc:
            log.warning("generation failed: %s", exc)
            return RAGAnswer(
                question=question,
                text=IDK_TEXT,
                abstained=True,
                abstain_reason=f"llm_error:{type(exc).__name__}",
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                retrieved=[r.to_dict() for r in result.retrieved[:8]],
                model=self.settings.llm_provider,
            )

        text = neutralize_chat_tokens(raw).strip()
        indices = [int(n) for group in _CITATION_RE.findall(text) for n in group.split(",") if n.strip().isdigit()]
        citations = self._citations_from(indices, result.kept)
        ok, reason, citations = self._verify(text, result.kept, citations, question=question)

        if not ok:
            log.info("abstaining (%s) for question hash=%s", reason, hash(search_query) & 0xFFFF)
            text = IDK_TEXT
        elif looks_like_refusal(text):
            # Normalise every refusal to the single canonical sentence.
            text = IDK_TEXT
            citations = []
        else:
            # A 3B model sometimes opens with a bare marker ("[1] The oil ...").
            text = re.sub(r"^\s*(?:\[\d+\]\s*)+", "", text).strip()

        return RAGAnswer(
            question=question,
            text=text,
            citations=citations if ok else [],
            abstained=(not ok) or looks_like_refusal(text),
            abstain_reason=reason,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            retrieved=[r.to_dict() for r in result.retrieved[:8]],
            reranker_fallback=result.reranker_fallback,
            model=self.settings.llm_provider,
        )

    def stream_answer(
        self,
        question: str,
        history: Sequence[dict[str, str]] | None = None,
    ) -> Iterator[tuple[str, Any]]:
        """Yield ``("token", str)`` deltas, then ``("result", RAGAnswer)``.

        The caller must be able to *replace* the streamed text, because
        verification runs on the full answer and may downgrade it to "I don't know."
        """
        started = time.perf_counter()
        question = clean_question(question, self.settings.max_question_chars)
        search_query = self._standalone_question(question, history)
        result = self._retrieve(search_query)

        if not result.kept:
            yield "result", RAGAnswer(
                question=question,
                text=IDK_TEXT,
                abstained=True,
                abstain_reason="no_relevant_context",
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                retrieved=[r.to_dict() for r in result.retrieved[:8]],
                model=self.settings.llm_provider,
            )
            return

        missing = self._unmatched_identifiers(question, result.kept)
        if missing:
            yield "result", RAGAnswer(
                question=question,
                text=IDK_TEXT,
                abstained=True,
                abstain_reason=f"unmatched_identifier:{','.join(missing)}",
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                retrieved=[r.to_dict() for r in result.retrieved[:8]],
                model=self.settings.llm_provider,
            )
            return

        context = self.retriever.build_context(result.kept)
        messages = [
            Message("system", SYSTEM_PROMPT),
            Message("user", build_user_prompt(question, context)),
        ]
        chunks: list[str] = []
        try:
            for delta in self.llm.stream(messages, temperature=self.settings.temperature):
                chunks.append(delta)
                yield "token", delta
        except LLMError:
            yield "result", RAGAnswer(
                question=question,
                text=IDK_TEXT,
                abstained=True,
                abstain_reason="llm_error",
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                retrieved=[r.to_dict() for r in result.retrieved[:8]],
                model=self.settings.llm_provider,
            )
            return

        raw = "".join(chunks)
        indices = [int(n) for group in _CITATION_RE.findall(raw) for n in group.split(",") if n.strip().isdigit()]
        citations = self._citations_from(indices, result.kept)
        ok, reason, citations = self._verify(raw, result.kept, citations, question=question)
        final_text = raw.strip() if ok else IDK_TEXT
        if ok and looks_like_refusal(raw):
            final_text = IDK_TEXT
            citations = []

        yield "result", RAGAnswer(
            question=question,
            text=final_text,
            citations=citations if ok else [],
            abstained=(not ok) or looks_like_refusal(raw),
            abstain_reason=reason,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            retrieved=[r.to_dict() for r in result.retrieved[:8]],
            reranker_fallback=result.reranker_fallback,
            model=self.settings.llm_provider,
        )

    # ------------------------------------------------- extractive CI baseline
    def answer_extractive(self, question: str, history: Sequence[dict[str, str]] | None = None) -> RAGAnswer:
        """No-LLM baseline used by CI/tests: returns the best supporting sentences.

        This is *not* the product answer path. It exists so the evaluation gate can
        run on a GitHub runner with no model server, and it exercises the same
        retrieval + abstention logic.
        """
        started = time.perf_counter()
        question = clean_question(question, self.settings.max_question_chars)
        result = self._retrieve(question)
        if not result.kept:
            return RAGAnswer(
                question=question,
                text=IDK_TEXT,
                abstained=True,
                abstain_reason="no_relevant_context",
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                mode="extractive",
                model="extractive-baseline",
            )
        missing = self._unmatched_identifiers(question, result.kept)
        if missing:
            return RAGAnswer(
                question=question,
                text=IDK_TEXT,
                abstained=True,
                abstain_reason=f"unmatched_identifier:{','.join(missing)}",
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                mode="extractive",
                model="extractive-baseline",
            )
        q_tokens = content_tokens(question)
        q_numbers = set(numbers_in(question))
        q_stems = {_stem(token) for token in q_tokens}
        q_years = {number for number in q_numbers if len(number) == 4 and number.isdigit()}

        # Scan every retrieved chunk, not just the top one: re-rank order is a
        # heuristic, and the answer often sits at rank 2-4.
        best: tuple[float, int, str] | None = None
        for index, item in enumerate(result.kept, start=1):
            body = item.text
            heading = str(item.metadata.get("heading") or "")
            if heading and body.startswith(heading):
                body = body[len(heading) :]
            for sentence in _candidate_parts(body):
                sentence = sentence.strip(" |-*")
                if len(sentence.split()) < 3 or len(sentence) > 900:
                    continue
                tokens = content_tokens(sentence)
                if not tokens:
                    continue
                overlap = len(q_stems & {_stem(token) for token in tokens}) / max(1, len(q_stems))
                # Prefer a sentence that also carries the number the question asks about.
                if q_numbers & set(numbers_in(sentence)):
                    overlap += 0.05
                # A sentence that talks about a different year is answering a
                # different question ("FY2025 guidance" vs "FY2024 margin").
                sentence_years = {n for n in numbers_in(sentence) if len(n) == 4 and n.isdigit()}
                if q_years and sentence_years and not (q_years & sentence_years):
                    overlap -= 0.35
                overlap += 0.08 / index
                if best is None or overlap > best[0]:
                    best = (overlap, index, sentence)

        if best is None or best[0] < 0.30:
            return RAGAnswer(
                question=question,
                text=IDK_TEXT,
                abstained=True,
                abstain_reason="no_extractive_support",
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                retrieved=[r.to_dict() for r in result.retrieved[:8]],
                mode="extractive",
                model="extractive-baseline",
            )
        score, index, sentence = best
        answer_text = f"{sentence} [{index}]"
        citations = self._citations_from([index], result.kept)
        ok, reason, citations = self._verify(answer_text, result.kept, citations, question=question)
        return RAGAnswer(
            question=question,
            text=answer_text if ok else IDK_TEXT,
            citations=citations if ok else [],
            abstained=not ok,
            abstain_reason=reason,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            retrieved=[r.to_dict() for r in result.retrieved[:8]],
            mode="extractive",
            model="extractive-baseline",
        )
