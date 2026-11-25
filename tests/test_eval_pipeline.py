from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragkit.eval.dataset import GoldenSample, load_golden_set, validate_golden_set
from ragkit.eval.metrics import ScoredRecord, abstention_accuracy, heuristic_scores
from ragkit.eval.report import build_report, evaluate_gate, to_markdown

ROOT = Path(__file__).resolve().parents[1]


def test_golden_set_has_twenty_questions():
    samples = load_golden_set(ROOT / "data" / "eval" / "golden_set.jsonl")
    assert len(samples) == 20
    assert sum(1 for s in samples if not s.answerable) == 4


def test_golden_set_is_valid():
    samples = load_golden_set(ROOT / "data" / "eval" / "golden_set.jsonl")
    assert validate_golden_set(samples) == []


def test_golden_set_rejects_missing_fields(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "x", "question": "What?"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_golden_set(path)


def test_duplicate_ids_rejected(tmp_path):
    path = tmp_path / "dupes.jsonl"
    row = {"id": "a", "question": "What is revenue?", "ground_truth": "1", "answerable": True}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_golden_set(path)


def _record(sample: GoldenSample, answer: str, abstained: bool, contexts: list[str], latency: float = 100.0) -> ScoredRecord:
    return ScoredRecord(
        sample=sample,
        answer=answer,
        contexts=contexts,
        citations=[],
        abstained=abstained,
        abstain_reason="" if not abstained else "no_relevant_context",
        latency_ms=latency,
        retrieved=[{"chunk_id": "c1", "source": "doc.md", "snippet": context, "score": 0.8} for context in contexts],
    )


def test_heuristic_scores_are_bounded_and_reward_grounding():
    sample = GoldenSample(
        id="a1",
        question="What was revenue in fiscal 2024?",
        ground_truth="Revenue was $4,812 million in fiscal 2024.",
        reference_contexts=["Revenue was $4,812 million in fiscal 2024."],
    )
    good = _record(sample, "Revenue was $4,812 million in fiscal 2024 [1].", False, ["Revenue was $4,812 million in fiscal 2024."])
    bad = _record(sample, "Revenue was $9,999 million in fiscal 2019.", False, ["The chiller uses R-513A refrigerant."])
    good_scores = heuristic_scores([good])
    bad_scores = heuristic_scores([bad])
    assert 0.0 <= good_scores["faithfulness"] <= 1.0
    assert good_scores["faithfulness"] > bad_scores["faithfulness"]
    assert good_scores["numeric_fidelity"] == 1.0
    assert bad_scores["numeric_fidelity"] == 0.0


def test_abstention_accuracy():
    answerable = GoldenSample(id="a", question="Q?", ground_truth="A", answerable=True, reference_contexts=["A"])
    control = GoldenSample(id="b", question="Q?", ground_truth="I don't know.", answerable=False)
    records = [
        _record(answerable, "A [1]", False, ["A"]),
        _record(control, "I don't know.", True, []),
    ]
    assert abstention_accuracy(records) == 1.0
    records[0] = _record(answerable, "I don't know.", True, [])
    assert abstention_accuracy(records) == 0.5


def test_gate_passes_on_strong_metrics():
    gate = evaluate_gate(
        {"context_recall": 0.82, "hit_rate": 0.95, "abstention_accuracy": 1.0, "numeric_fidelity": 1.0},
        "heuristic",
    )
    assert gate.passed, gate.failures


def test_gate_fails_below_floor():
    gate = evaluate_gate(
        {"context_recall": 0.20, "hit_rate": 0.30, "abstention_accuracy": 0.50, "numeric_fidelity": 0.40},
        "heuristic",
    )
    assert not gate.passed
    assert any("context_recall" in f for f in gate.failures)


def test_gate_detects_regression_against_baseline():
    metrics = {"context_recall": 0.70, "hit_rate": 0.85, "abstention_accuracy": 0.90, "numeric_fidelity": 0.95}
    gate = evaluate_gate(metrics, "heuristic", baseline={"context_recall": 0.90}, tolerance=0.05)
    assert not gate.passed
    assert any("regression" in f for f in gate.failures)


def test_gate_tolerates_small_noise():
    metrics = {"context_recall": 0.87, "hit_rate": 0.85, "abstention_accuracy": 0.90, "numeric_fidelity": 0.95}
    gate = evaluate_gate(metrics, "heuristic", baseline={"context_recall": 0.90}, tolerance=0.05)
    assert gate.passed, gate.failures


def test_report_renders_markdown(tmp_settings):
    sample = GoldenSample(id="a1", question="Q?", ground_truth="A", reference_contexts=["A"])
    records = [_record(sample, "A [1]", False, ["A"])]
    metrics = heuristic_scores(records)
    gate = evaluate_gate(metrics, "heuristic")
    report = build_report(records, metrics, "heuristic", tmp_settings, "llm", gate)
    markdown = to_markdown(report)
    assert "# RAG Evaluation Report" in markdown
    assert "| `hit_rate` |" in markdown
    assert "PASS" in markdown or "FAIL" in markdown
