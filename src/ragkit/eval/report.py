"""Report generation + the regression gate.

The gate is the whole point of this project: numbers are compared to a committed
baseline, and a drop of more than ``REGRESSION_TOLERANCE`` fails CI. Quality can
go up, but it cannot quietly go down.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..config import Settings
from ..logging_utils import get_logger
from .dataset import GoldenSample
from .metrics import ScoredRecord

log = get_logger("eval.report")

SCHEMA_VERSION = 1
REGRESSION_TOLERANCE = 0.05

# Absolute floors. A pipeline below these is not shippable regardless of baseline.
GATE_FLOORS: dict[str, dict[str, float]] = {
    "ragas": {
        "faithfulness": 0.70,
        "answer_relevancy": 0.55,
        "context_recall": 0.60,
        "abstention_accuracy": 0.85,
    },
    "heuristic": {
        "context_recall": 0.50,
        "hit_rate": 0.70,
        "abstention_accuracy": 0.85,
        "numeric_fidelity": 0.90,
    },
}

# Metrics to compare against the baseline, per judge.
COMPARABLE = {
    "ragas": ["faithfulness", "answer_relevancy", "context_recall", "abstention_accuracy", "hit_rate"],
    "heuristic": ["faithfulness", "context_recall", "hit_rate", "mrr", "abstention_accuracy", "numeric_fidelity"],
}


def git_sha() -> str:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"][:12]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class GateResult:
    passed: bool
    failures: list[str]
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "failures": self.failures, "mode": self.mode}


def evaluate_gate(
    metrics: dict[str, float],
    mode: str,
    baseline: dict[str, float] | None = None,
    tolerance: float = REGRESSION_TOLERANCE,
) -> GateResult:
    failures: list[str] = []
    floors = GATE_FLOORS.get(mode, GATE_FLOORS["heuristic"])
    for name, floor in floors.items():
        value = metrics.get(name)
        if value is None:
            failures.append(f"{name}: missing from report (expected >= {floor:.2f})")
        elif value < floor:
            failures.append(f"{name}: {value:.3f} < floor {floor:.2f}")

    if baseline:
        for name in COMPARABLE.get(mode, []):
            current = metrics.get(name)
            previous = baseline.get(name)
            if current is None or previous is None:
                continue
            if current < previous - tolerance:
                failures.append(
                    f"{name}: regression {previous:.3f} -> {current:.3f} (tolerance {tolerance:.2f})"
                )
    return GateResult(passed=not failures, failures=failures, mode=mode)


def load_baseline(settings: Settings, judge: str | None = None) -> dict[str, float]:
    """Load the committed baseline.

    A baseline only means something for the judge that produced it: Ragas and
    heuristic scores are not on the same scale, so a mismatch returns ``{}``.
    """
    path = settings.eval_dir / "baseline.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if judge and payload.get("judge") and payload["judge"] != judge:
            log.warning("baseline was produced by '%s', not '%s'; ignoring it", payload["judge"], judge)
            return {}
        return {k: float(v) for k, v in payload.get("metrics", {}).items()}
    except (json.JSONDecodeError, OSError, ValueError):
        return {}


def build_report(
    records: Sequence[ScoredRecord],
    metrics: dict[str, float],
    judge: str,
    settings: Settings,
    mode: str,
    gate: GateResult,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "git_sha": git_sha(),
        "judge": judge,
        "answer_mode": mode,
        "metrics": metrics,
        "gate": gate.to_dict(),
        "notes": list(notes),
        "config": {
            "llm_provider": settings.llm_provider,
            "llm_model": settings.ollama_model if settings.llm_provider == "ollama" else settings.openai_model,
            "embed_model": settings.embed_model,
            "rerank_model": settings.rerank_model,
            "chunk_max_chars": settings.chunk_max_chars,
            "semantic_percentile": settings.semantic_percentile,
            "top_k_final": settings.top_k_final,
            "rerank_min_score": settings.rerank_min_score,
            "grounding_min_overlap": settings.grounding_min_overlap,
            "vector_backend": settings.vector_backend,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "ci": bool(os.getenv("CI")),
        },
        "records": [record.to_dict() for record in records],
    }


def write_report(report: dict[str, Any], settings: Settings) -> dict[str, Path]:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    settings.history_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    latest = settings.reports_dir / "latest.json"
    latest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    snapshot = settings.reports_dir / f"report-{stamp}.json"
    snapshot.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    history = settings.history_dir / f"{stamp}-{report['git_sha']}.jsonl"
    with history.open("w", encoding="utf-8") as fh:
        for record in report["records"]:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    markdown = settings.reports_dir / "latest.md"
    markdown.write_text(to_markdown(report), encoding="utf-8")

    summary = settings.reports_dir / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "generated_at": report["generated_at"],
                "git_sha": report["git_sha"],
                "judge": report["judge"],
                "metrics": report["metrics"],
                "gate": report["gate"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"latest": latest, "snapshot": snapshot, "history": history, "markdown": markdown}


def write_baseline(report: dict[str, Any], settings: Settings) -> Path:
    path = settings.eval_dir / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "updated_at": report["generated_at"],
                "git_sha": report["git_sha"],
                "judge": report["judge"],
                "answer_mode": report["answer_mode"],
                "metrics": report["metrics"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def to_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    gate = report["gate"]
    rows = "\n".join(f"| `{name}` | {_fmt(value)} |" for name, value in metrics.items())
    status = "✅ PASS" if gate["passed"] else "❌ FAIL"
    failures = "\n".join(f"- {f}" for f in gate["failures"]) or "- none"

    detail_rows = []
    for record in report["records"]:
        answer = " ".join((record["answer"] or "").split())[:160]
        detail_rows.append(
            f"| `{record['id']}` | {'abstained' if record['abstained'] else 'answered'} | "
            f"{record['latency_ms']:.0f} | {answer.replace('|', '/')} |"
        )
    detail = "\n".join(detail_rows)

    notes = "\n".join(f"- {n}" for n in report.get("notes", [])) or "- none"

    return f"""# RAG Evaluation Report

**Generated:** {report['generated_at']}
**Commit:** `{report['git_sha']}`
**Judge:** `{report['judge']}` (answer mode: `{report['answer_mode']}`)
**Gate:** {status}

## Metrics

| Metric | Score |
| --- | --- |
{rows}

## Gate

{status}

{failures}

## Configuration

| Setting | Value |
| --- | --- |
| LLM | `{report['config']['llm_provider']}` / `{report['config']['llm_model']}` |
| Embeddings | `{report['config']['embed_model']}` |
| Re-ranker | `{report['config']['rerank_model']}` |
| Chunk size | {report['config']['chunk_max_chars']} chars (semantic p{report['config']['semantic_percentile']:.0f}) |
| Final top-k | {report['config']['top_k_final']} |
| Python | {report['environment']['python']} |

## Notes

{notes}

## Per-question results

| # | Behaviour | Latency (ms) | Answer (truncated) |
| --- | --- | --- | --- |
{detail}
"""
