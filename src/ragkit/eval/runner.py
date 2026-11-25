"""Execute the pipeline over the golden set and collect per-question evidence."""

from __future__ import annotations

import time
from typing import Callable, Sequence

from ..logging_utils import get_logger, sha256_text
from ..rag import RAGPipeline
from .dataset import GoldenSample
from .metrics import ScoredRecord

log = get_logger("eval.runner")


def run_samples(
    pipeline: RAGPipeline,
    samples: Sequence[GoldenSample],
    mode: str = "llm",
    progress: Callable[[int, int, str], None] | None = None,
) -> list[ScoredRecord]:
    """Run each golden question through the pipeline.

    ``mode="extractive"`` uses the deterministic no-LLM baseline (CI-safe).
    """
    records: list[ScoredRecord] = []
    total = len(samples)
    for i, sample in enumerate(samples, start=1):
        started = time.perf_counter()
        error = ""
        try:
            if mode == "extractive":
                answer = pipeline.answer_extractive(sample.question)
            else:
                answer = pipeline.answer(sample.question)
            # use the full retrieved text, not only an answer snippet
            contexts = [item.get("text", "") for item in answer.retrieved[: pipeline.settings.top_k_final]]
            record = ScoredRecord(
                sample=sample,
                answer=answer.text,
                contexts=[c for c in contexts if c],
                citations=[c.to_dict() for c in answer.citations],
                abstained=answer.abstained,
                abstain_reason=answer.abstain_reason,
                latency_ms=answer.latency_ms,
                retrieved=answer.retrieved,
            )
        except Exception as exc:  # a single bad question must not abort the run
            log.warning("sample %s failed: %s", sample.id, type(exc).__name__)
            error = f"{type(exc).__name__}: {exc}"
            record = ScoredRecord(
                sample=sample,
                answer="",
                contexts=[],
                citations=[],
                abstained=True,
                abstain_reason="runner_error",
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                retrieved=[],
                error=error,
            )
        records.append(record)
        log.info(
            "[%d/%d] %s took %.0f ms (%s)",
            i,
            total,
            sample.id,
            record.latency_ms,
            "abstained" if record.abstained else "answered",
        )
        if progress:
            try:
                progress(i, total, sample.id)
            except Exception:  # pragma: no cover
                pass
    return records


def question_fingerprint(question: str) -> str:
    return sha256_text(question)[:12]
