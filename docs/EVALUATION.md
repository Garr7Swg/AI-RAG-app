# Evaluation

Evaluation is the difference between "it answered something" and "it answered correctly and
will keep doing so". This document explains the dataset, the metrics, the gate, and how to
extend them.

## 1. The golden set

`data/eval/golden_set.jsonl` — one JSON object per line:

```json
{
  "id": "q01",
  "question": "What was Northwind's total revenue in fiscal 2024 and how much did it grow year over year?",
  "ground_truth": "Revenue was $4,812 million, an increase of 7.4% from $4,480 million in fiscal 2023.",
  "reference_contexts": ["Consolidated revenue for fiscal year 2024 was $4,812 million, ..."],
  "answerable": true,
  "tags": ["financials", "multi-fact"],
  "difficulty": "easy"
}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | unique; appears in every report row |
| `question` | yes | sent to the pipeline verbatim |
| `ground_truth` | yes | reference answer for Ragas and lexical recall |
| `reference_contexts` | when answerable | the passages that contain the answer (used by hit-rate/MRR) |
| `answerable` | yes | `false` for refusal controls |
| `tags`, `difficulty` | no | grouping in reports |

Validation (`validate_golden_set`) enforces ≥20 samples, unique ids, reference contexts on
answerable items, an "I don't know" ground truth on controls, and at least one control.

### Why these 20

| Coverage | Questions |
| --- | --- |
| Single fact | revenue, adjusted EBITDA margin, diluted EPS, revolver size, refrigerant charge |
| Exact identifier / table lookup | alarm code E-41, oil interval, bearing torque, auditor name |
| Multi-hop across documents | leverage 1.61x vs covenant 3.50x |
| Numeric precision | 4.2 K approach alarm, 0.52 kW/ton, $5,050-5,200 M guidance |
| Refusal controls | Q3 FY2025 revenue (undisclosed), CEO compensation (not in corpus), CX-1200 (wrong model), UK prime minister (out of domain) |

The four controls matter most: a pipeline that answers everything scores well on faithfulness
but is useless in production. `abstention_accuracy` is scored across all 20 questions.

## 2. Metrics

### Ragas (primary)

| Metric | Question it answers | Needs |
| --- | --- | --- |
| **Faithfulness** | Are the answer's claims supported by the retrieved context? | judge LLM |
| **Answer relevancy** | Does the answer address the question asked? | judge LLM + embeddings |
| **Context recall** | Did retrieval bring back the evidence needed for the ground truth? | judge LLM + reference |

Run: `python scripts/evaluate.py --judge ragas` (needs Ollama or `RAG_OPENAI_*`).
Only answered, answerable samples are scored by Ragas; controls are scored by
`abstention_accuracy` instead, because reference-based metrics are undefined for them.

### Deterministic (always computed)

| Metric | Definition |
| --- | --- |
| `hit_rate` | share of answerable questions where a retrieved chunk covers ≥45% of reference-context tokens or all ground-truth numbers |
| `mrr` | mean reciprocal rank of the first supporting chunk |
| `context_precision` | share of retrieved contexts relevant to the reference |
| `numeric_fidelity` | share of numbers in the answer present in the context |
| `abstention_accuracy` | answered iff answerable, for all 20 |
| `mean_latency_ms` | wall clock per answer |

### Why a deterministic fallback exists

CI runners have no GPU, no Ollama, and often no API key. A judge-free evaluator still catches
the two regressions that matter most — broken retrieval and broken abstention — because it
measures evidence, not prose. It is labelled `judge: heuristic` in every report so no one
mistakes it for Ragas.

The `--mode extractive` flag additionally replaces generation with a deterministic extractive
answerer. That makes the whole eval run reproducible with zero external services. Faithfulness
under that mode measures *extractive grounding*, not generative quality — stated explicitly in
the report's `notes`.

## 3. The gate

`src/ragkit/eval/report.py`:

```python
GATE_FLOORS = {
  "ragas":     {faithfulness: 0.70, answer_relevancy: 0.55, context_recall: 0.60, abstention_accuracy: 0.85},
  "heuristic": {context_recall: 0.50, hit_rate: 0.70, abstention_accuracy: 0.85, numeric_fidelity: 0.90},
}
REGRESSION_TOLERANCE = 0.05
```

Rules:

1. Every floor in the active judge's table must be met.
2. If a baseline exists for that judge, no comparable metric may fall more than `0.05` below it.
3. Ragas and heuristic numbers are never compared to each other (different scales).

Exit codes: `0` pass, `1` gate failure, `2` could not run (missing index/dataset).

## 4. Promoting a baseline

```bash
python scripts/evaluate.py --judge ragas --update-baseline
```

This writes `eval/baseline.json`. Review the diff before committing — a baseline is a promise.
In GitHub, dispatch **RAG Evaluation** with `update_baseline=true` to do it from CI.

Until a baseline exists, only the absolute floors apply, which is safe (they are the minimum
shippable quality) but does not detect slow drift.

## 5. Reading a report

`eval/reports/latest.md`:

```
## Metrics          faithfulness 0.83 / answer_relevancy 0.71 / context_recall 0.88 ...
## Gate             ✅ PASS
## Configuration    model, embedder, chunking, top-k, thresholds
## Per-question     id, answered/abstained, latency, truncated answer
```

`eval/reports/latest.json` adds, per question: full answer, retrieved chunks with scores,
citations, abstain reason, and the config used. That is enough to debug a regression without
re-running anything.

History: `eval/reports/history/<timestamp>-<sha>.jsonl`, committed by CI, one line per question.
`git log eval/reports` is the quality trend.

## 6. Adding questions

1. Append a line to `data/eval/golden_set.jsonl`, copying `reference_contexts` verbatim from the
   corpus (this keeps retrieval metrics meaningful).
2. Run `python -m pytest tests/test_eval_pipeline.py` — validation must pass.
3. Run `python scripts/evaluate.py --judge heuristic` and check the new question is answered.
4. If it legitimately changes the achievable score, update the baseline deliberately.

Good additions: questions whose answer spans two documents, questions with a plausible distractor
in the corpus, and refusals for topics adjacent to your documents (the hardest controls).

## 7. What is *not* measured

- Multi-turn conversation quality (the pipeline supports history but the golden set is single-turn).
- Latency SLOs under load (the metrics are single-user wall clock).
- Retrieval fairness across document types — 3 documents is enough for a demo, not for a
  production corpus of thousands.
- Cost per answer (hosted provider billing is out of scope).

Being explicit about this is part of the deliverable: an evaluation suite that overstates its
coverage is worse than a narrow one that is honest.

## 8. Reference runs

Both runs below were measured on the bundled 3-document corpus (40 chunks after ingestion),
Windows / Python 3.13, 20 questions, 2026-11-21. They are two different judges and must not be
compared metric-by-metric; Ragas mode is the primary gate and the extractive run is the
no-model baseline CI falls back to.

### 8.1 Ragas judge, LLM answers (`--judge ragas --mode llm`)

Local Ollama (`qwen2.5:3b-instruct`) answering, and the same model acting as Ragas judge and
embeddings, so no paid API key is involved. Gate **PASS**. 15 samples were scored (the 5
abstentions have no answer/contexts to judge and are scored by `abstention_accuracy` instead):

| Metric | Value |
| --- | --- |
| `faithfulness` | 0.733 |
| `answer_relevancy` | 0.912 |
| `context_recall` | 1.000 |
| `context_precision` | 0.438 |
| `hit_rate` | 1.000 |
| `mrr` | 1.000 |
| `numeric_fidelity` | 1.000 |
| `abstention_accuracy` | 0.950 |
| `answer_rate` | 0.750 |
| `mean_latency_ms` | ~4.3 s warm, ~23 s first call (model load) |

### 8.2 Deterministic baseline (`--judge heuristic --mode extractive`)

No model at all: answers are assembled from retrieved sentences, so these numbers isolate
retrieval and the guardrail layer from generation quality.

| Metric | Value |
| --- | --- |
| `context_recall` | 0.943 |
| `hit_rate` | 1.000 |
| `mrr` | 1.000 |
| `context_precision` | 0.438 |
| `numeric_fidelity` | 1.000 |
| `abstention_accuracy` | 0.900 |
| `mean_latency_ms` | ~1.8 s (first query ~14 s: model warm-up) |

`eval/baseline.json` holds the promoted Ragas run, so later Ragas runs are diffed against it
(the comparison is refused across judges).

Per-question behaviour worth knowing before you tune anything:

- **q07** ("Q3 FY2025 revenue") is refused in Ragas mode with `unsupported_number:2024`: the model
  drafted an answer containing a year that is not in the cited chunks, and the numeric-grounding
  guard replaced it with the canonical "I don't know". That is the guardrail working; the cost is
  one abandoned sample, part of why `answer_rate` is 0.75.
- **q17** ("Q3 FY2025 revenue") is refused by the model itself (`model_abstained`) — the only
  quarter the corpus reports is fiscal 2024, so there is no defensible answer. In extractive
  mode this question is answered instead, because lexical overlap cannot tell "fiscal 2024
  revenue" from "Q3 2025 revenue"; this is exactly why the refusal controls exist and why Ragas
  mode is the primary judge.
- **q08** ("cross-default threshold") is answered correctly by the LLM, but abstains in
  extractive mode: the corpus never uses the word *threshold* in Section 8.3, so overlap falls
  just under the extractive floor while the chunk is still retrieved.
- **q19** ("CX-1200") abstains via the identifier guard (`unmatched_identifier:cx-1200`):
  `cx-1200` appears nowhere in the evidence, so the system refuses instead of answering about
  the CX-900.
- **q20** is deliberately out of domain and abstains in both judges.

`context_precision` is structurally low for this corpus because one 1400-character chunk often
contains the answer plus three unrelated facts; it is a chunking-granularity signal, not a bug.
