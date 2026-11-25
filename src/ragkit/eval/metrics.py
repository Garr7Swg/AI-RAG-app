"""Metrics.

Two evaluators behind one interface:

``ragas``     - the real thing. Faithfulness, Answer Relevancy, Context Recall
                judged by an LLM (local Ollama or any OpenAI-compatible model).
``heuristic`` - a deterministic, offline fallback used by CI when no model server
                exists. It is *not* a replacement for Ragas; it scores lexical
                grounding and retrieval coverage, so a broken retriever or a
                broken guardrail still fails the build.

Retrieval metrics (hit rate, MRR, context precision) are always computed, because
they are free and catch most regressions before the judge even runs.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..config import Settings
from ..logging_utils import get_logger
from ..rag import content_tokens, numbers_in
from .dataset import GoldenSample

log = get_logger("eval.metrics")

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’.-]*")


class EvalUnavailable(RuntimeError):
    """Raised when Ragas (or its judge model) cannot run."""


@dataclass
class ScoredRecord:
    sample: GoldenSample
    answer: str
    contexts: list[str]
    citations: list[dict[str, Any]]
    abstained: bool
    abstain_reason: str
    latency_ms: float
    retrieved: list[dict[str, Any]]
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.sample.id,
            "question": self.sample.question,
            "ground_truth": self.sample.ground_truth,
            "answerable": self.sample.answerable,
            "tags": self.sample.tags,
            "difficulty": self.sample.difficulty,
            "answer": self.answer,
            "contexts": self.contexts,
            "citations": self.citations,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "latency_ms": self.latency_ms,
            "retrieved": [
                {
                    "chunk_id": r.get("chunk_id"),
                    "source": r.get("source") or r.get("metadata", {}).get("source"),
                    "score": r.get("score"),
                    "snippet": (r.get("snippet") or "")[:240],
                }
                for r in self.retrieved
            ],
            "error": self.error,
        }


# --------------------------------------------------------------- helpers
def _overlap(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    if not a:
        return 0.0
    return len(a & b) / len(a)


def _sentence_support(answer: str, context: str) -> float:
    """Fraction of answer sentences whose content words appear in the context."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", answer or "") if len(s.split()) >= 3]
    if not sentences:
        return 0.0
    context_tokens = content_tokens(context)
    supported = 0
    for sentence in sentences:
        tokens = content_tokens(sentence)
        if not tokens:
            continue
        if len(tokens & context_tokens) / len(tokens) >= 0.55:
            supported += 1
    return supported / len(sentences)


def _numeric_fidelity(answer: str, context: str) -> float:
    """Share of numbers in the answer that also appear in the context."""
    answer_numbers = numbers_in(answer)
    if not answer_numbers:
        return 1.0
    context_text = (context or "").replace(",", "")
    hits = sum(1 for n in answer_numbers if n.replace(",", "") in context_text)
    return hits / len(answer_numbers)


def retrieval_metrics(record: ScoredRecord) -> dict[str, float]:
    """Hit-rate / MRR using the sample's reference contexts and ground truth."""
    if not record.sample.answerable or not record.sample.reference_contexts:
        return {}
    reference_tokens = content_tokens(" ".join(record.sample.reference_contexts))
    truth_numbers = set(numbers_in(record.sample.ground_truth))
    rank_hit = None
    for rank, item in enumerate(record.retrieved, start=1):
        text = str(item.get("text") or item.get("snippet") or "")
        tokens = content_tokens(text)
        token_hit = bool(tokens) and len(tokens & reference_tokens) / max(1, len(reference_tokens)) >= 0.45
        number_hit = bool(truth_numbers) and truth_numbers.issubset(set(numbers_in(text)))
        if token_hit or number_hit:
            rank_hit = rank
            break
    hit = 1.0 if rank_hit else 0.0
    return {
        "hit": hit,
        "mrr": (1.0 / rank_hit) if rank_hit else 0.0,
        "context_precision": _context_precision(record, reference_tokens),
    }


def _context_precision(record: ScoredRecord, reference_tokens: set[str]) -> float:
    if not record.contexts:
        return 0.0
    relevant = 0
    for context in record.contexts:
        tokens = content_tokens(context)
        if tokens and len(tokens & reference_tokens) / max(1, len(reference_tokens)) >= 0.30:
            relevant += 1
    return relevant / len(record.contexts)


def abstention_accuracy(records: Sequence[ScoredRecord]) -> float:
    """Did the pipeline refuse when it should, and answer when it should?"""
    if not records:
        return 0.0
    correct = 0
    for record in records:
        should_abstain = not record.sample.answerable
        if record.abstained == should_abstain:
            correct += 1
    return correct / len(records)


# --------------------------------------------------------------- heuristic
def heuristic_scores(records: Sequence[ScoredRecord]) -> dict[str, float]:
    answerable = [r for r in records if r.sample.answerable and not r.abstained]
    answered_share = round(len(answerable) / len(records), 4) if records else 0.0
    faithfulness: list[float] = []
    relevancy: list[float] = []
    recall: list[float] = []
    fidelity: list[float] = []

    for record in answerable:
        context = "\n".join(record.contexts)
        faithfulness.append(_sentence_support(record.answer, context))
        question_tokens = content_tokens(record.sample.question)
        relevancy.append(min(1.0, _overlap(question_tokens, content_tokens(record.answer)) * 1.4))
        truth_tokens = content_tokens(record.sample.ground_truth)
        recall.append(min(1.0, _overlap(truth_tokens, content_tokens(context)) * 1.15))
        fidelity.append(_numeric_fidelity(record.answer, context))

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    retrieval = [retrieval_metrics(r) for r in records]
    hits = [m["hit"] for m in retrieval if m]
    mrrs = [m["mrr"] for m in retrieval if m]
    precisions = [m["context_precision"] for m in retrieval if m]

    return {
        "faithfulness": mean(faithfulness),
        "answer_relevancy": mean(relevancy),
        "context_recall": mean(recall),
        "numeric_fidelity": mean(fidelity),
        "hit_rate": round(sum(hits) / len(hits), 4) if hits else 0.0,
        "mrr": round(sum(mrrs) / len(mrrs), 4) if mrrs else 0.0,
        "context_precision": round(sum(precisions) / len(precisions), 4) if precisions else 0.0,
        "abstention_accuracy": round(abstention_accuracy(records), 4),
        "answer_rate": answered_share,
        "mean_latency_ms": round(sum(r.latency_ms for r in records) / len(records), 1) if records else 0.0,
    }


# ------------------------------------------------------------------- ragas
def _ragas_embeddings(embedder: Any):
    """Adapt our sentence-transformers embedder to LangChain's Embeddings API."""
    try:
        from langchain_core.embeddings import Embeddings
    except Exception as exc:  # pragma: no cover
        raise EvalUnavailable("langchain-core is required for Ragas embeddings.") from exc

    class RagasEmbeddings(Embeddings):
        # Closure over `embedder`: works whether Embeddings is an ABC (current
        # LangChain) or a pydantic model (older), with no __init__ subtleties.
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return embedder.encode(list(texts)).tolist()

        def embed_query(self, text: str) -> list[float]:
            return embedder.encode_one(text, is_query=True).tolist()

    return RagasEmbeddings()


def ragas_scores(
    records: Sequence[ScoredRecord],
    settings: Settings,
    embedder: Any | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Run Ragas on the answerable subset. Returns (metrics, meta)."""
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy
    except Exception as exc:
        raise EvalUnavailable(f"Ragas is not installed: {type(exc).__name__}") from exc

    from ..llm import get_chat_model

    try:
        judge = get_chat_model(settings, temperature=0.0)
    except Exception as exc:
        raise EvalUnavailable(f"Judge model unavailable: {type(exc).__name__}") from exc

    answerable = [r for r in records if r.sample.answerable and not r.abstained and not r.error]
    if not answerable:
        raise EvalUnavailable("No answered samples to score with Ragas.")

    embeddings = None
    if embedder is not None:
        try:
            embeddings = _ragas_embeddings(embedder)
        except Exception:  # pragma: no cover
            embeddings = None

    # Pass the judge and embeddings explicitly. Without this, newer Ragas
    # releases build a default OpenAI client and demand OPENAI_API_KEY even
    # though we are running fully locally.
    try:
        metrics = [
            Faithfulness(llm=judge),
            ResponseRelevancy(llm=judge, embeddings=embeddings),
            LLMContextRecall(llm=judge),
        ]
    except TypeError:  # older signature
        metrics = [Faithfulness(), ResponseRelevancy(), LLMContextRecall()]
        for metric in metrics:
            try:
                metric.llm = judge
                if embeddings is not None and hasattr(metric, "embeddings"):
                    metric.embeddings = embeddings
            except Exception:  # pragma: no cover - version drift
                pass

    samples = [
        SingleTurnSample(
            user_input=r.sample.question,
            response=r.answer,
            retrieved_contexts=r.contexts,
            reference=r.sample.ground_truth,
            reference_contexts=r.sample.reference_contexts or None,
        )
        for r in answerable
    ]
    dataset = EvaluationDataset(samples=samples)

    try:
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "metrics": metrics,
            "raise_exceptions": False,
            "show_progress": False,
            "llm": judge,
        }
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        try:
            result = evaluate(**kwargs)
        except TypeError:  # some versions do not accept embeddings=
            kwargs.pop("embeddings", None)
            result = evaluate(**kwargs)
    except Exception as exc:
        raise EvalUnavailable(f"Ragas evaluation failed: {type(exc).__name__}: {exc}") from exc

    frame = result.to_pandas()
    scores: dict[str, float] = {}
    aliases = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "response_relevancy": "answer_relevancy",
        "context_recall": "context_recall",
        "llm_context_recall": "context_recall",
    }
    for column in frame.columns:
        key = aliases.get(str(column).lower())
        if not key:
            continue
        values = [float(v) for v in frame[column].tolist() if v is not None and not math.isnan(float(v))]
        if values:
            scores[key] = round(sum(values) / len(values), 4)

    if not scores:
        raise EvalUnavailable("Ragas returned no usable scores.")

    meta = {
        "samples_scored": len(answerable),
        "judge": settings.ollama_model if settings.llm_provider == "ollama" else settings.openai_model,
    }
    return scores, meta
