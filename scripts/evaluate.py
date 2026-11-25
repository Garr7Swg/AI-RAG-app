#!/usr/bin/env python
"""CLI: evaluate the pipeline against the 20-question golden set.

    python scripts/evaluate.py --judge ragas       # LLM judge (Ollama or API)
    python scripts/evaluate.py --judge heuristic   # offline, CI-safe
    python scripts/evaluate.py --judge ragas --update-baseline

Exit codes: 0 = gate passed, 1 = gate failed, 2 = could not run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.config import get_settings  # noqa: E402
from ragkit.eval.dataset import load_golden_set, validate_golden_set  # noqa: E402
from ragkit.eval.metrics import EvalUnavailable, heuristic_scores, ragas_scores  # noqa: E402
from ragkit.eval.report import (  # noqa: E402
    build_report,
    evaluate_gate,
    load_baseline,
    to_markdown,
    write_baseline,
    write_report,
)
from ragkit.eval.runner import run_samples  # noqa: E402
from ragkit.ingest import load_manifest  # noqa: E402
from ragkit.logging_utils import configure_logging, get_logger, redact  # noqa: E402
from ragkit.rag import RAGPipeline  # noqa: E402

log = get_logger("cli.eval")


def _force_utf8_console() -> None:
    """Windows consoles default to cp1252 and die on the check marks in the report."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover - non-standard streams
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the RAG pipeline on the golden set.")
    parser.add_argument("--judge", choices=["ragas", "heuristic"], default="heuristic")
    parser.add_argument("--mode", choices=["llm", "extractive"], default="llm",
                        help="extractive = deterministic baseline for CI (no LLM)")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N questions")
    parser.add_argument("--update-baseline", action="store_true", help="Overwrite eval/baseline.json")
    parser.add_argument("--no-gate", action="store_true", help="Report only, never fail")
    parser.add_argument("--json", action="store_true", help="Print the full JSON report")
    args = parser.parse_args()
    _force_utf8_console()
    configure_logging()

    settings = get_settings()

    try:
        samples = load_golden_set(settings.golden_set_path)
    except (FileNotFoundError, ValueError) as exc:
        log.error("golden set problem: %s", exc)
        return 2

    problems = validate_golden_set(samples)
    for problem in problems:
        log.warning("golden set: %s", problem)
    if args.limit:
        samples = samples[: args.limit]

    manifest = load_manifest(settings)
    if not manifest.get("chunk_count"):
        log.error("Index is empty. Run `make ingest` first.")
        return 2

    pipeline = RAGPipeline(settings)
    notes: list[str] = []
    if args.mode == "extractive":
        notes.append(
            "Answer mode is `extractive`: a deterministic no-LLM baseline is used so the gate can run "
            "on a CI runner without a model server. Faithfulness here reflects extractive grounding, "
            "not generative quality."
        )

    log.info("running %d questions (mode=%s, judge=%s)", len(samples), args.mode, args.judge)
    records = run_samples(pipeline, samples, mode=args.mode)

    metrics = heuristic_scores(records)
    judge_used = "heuristic"
    if args.judge == "ragas":
        try:
            ragas_metrics, meta = ragas_scores(records, settings, embedder=pipeline.embedder)
            metrics.update(ragas_metrics)
            judge_used = "ragas"
            notes.append(f"Ragas scored {meta['samples_scored']} answered samples with `{meta['judge']}`.")
        except EvalUnavailable as exc:
            notes.append(f"Ragas unavailable ({redact(str(exc), 200)}); fell back to the heuristic judge.")
            log.warning("Ragas unavailable: %s", exc)

    baseline = load_baseline(settings, judge=judge_used)
    if baseline and args.judge == "ragas" and judge_used == "heuristic":
        baseline = {}  # never compare across judges
    gate = evaluate_gate(metrics, judge_used, baseline)
    if args.no_gate:
        gate.passed = True
        gate.failures = []

    report = build_report(records, metrics, judge_used, settings, args.mode, gate, notes)
    paths = write_report(report, settings)
    if args.update_baseline:
        path = write_baseline(report, settings)
        print(f"baseline updated: {path}")

    print(to_markdown(report))
    if args.json:
        print(json.dumps(report, indent=2)[:20000])

    print(f"\nartefacts: {paths['latest']} | {paths['markdown']} | {paths['history']}")
    if gate.passed:
        print("GATE: PASS")
        return 0
    print("GATE: FAIL")
    for failure in gate.failures:
        print(f"  - {failure}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
