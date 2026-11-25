# RAG Evaluation Report

**Generated:** 2026-11-21T15:09:12+00:00
**Commit:** `unknown`
**Judge:** `ragas` (answer mode: `llm`)
**Gate:** ✅ PASS

## Metrics

| Metric | Score |
| --- | --- |
| `faithfulness` | 0.733 |
| `answer_relevancy` | 0.912 |
| `context_recall` | 1.000 |
| `numeric_fidelity` | 1.000 |
| `hit_rate` | 1.000 |
| `mrr` | 1.000 |
| `context_precision` | 0.438 |
| `abstention_accuracy` | 0.950 |
| `answer_rate` | 0.750 |
| `mean_latency_ms` | 4313.400 |

## Gate

✅ PASS

- none

## Configuration

| Setting | Value |
| --- | --- |
| LLM | `ollama` / `qwen2.5:3b-instruct` |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Chunk size | 1400 chars (semantic p88) |
| Final top-k | 5 |
| Python | 3.13.17 |

## Notes

- Ragas scored 15 answered samples with `qwen2.5:3b-instruct`.

## Per-question results

| # | Behaviour | Latency (ms) | Answer (truncated) |
| --- | --- | --- | --- |
| `q01` | answered | 22856 | Northwind's total revenue in fiscal 2024 was $4,812 million, which grew by 7.4% from $4,480 million in fiscal year 2023. [2] |
| `q02` | answered | 3650 | For fiscal year 2024, the adjusted EBITDA margin was 17.5%, as shown in [2]'s Financial Highlights section. |
| `q03` | answered | 3730 | In fiscal 2024, diluted earnings per share were $2.41 [1]. |
| `q04` | answered | 3903 | Northwind's actual consolidated net leverage ratio is 1.61 to 1.00, which is compliant with the maximum allowed by its credit agreement, which is 3.50 to 1.00.  |
| `q05` | answered | 3821 | The minimum consolidated interest coverage ratio the borrower must maintain is 2.75 to 1.00. [1] |
| `q06` | answered | 3698 | The revolving credit facility is $1,250 million and it matures on June 30, 2031. [1][2] |
| `q07` | abstained | 4758 | I don't know. The provided documents do not contain enough information to answer this question. |
| `q08` | answered | 3452 | The cross-default threshold in the credit agreement is an aggregate principal amount exceeding $50,000,000. |
| `q09` | answered | 3837 | The CX-900 uses R-513A as its refrigerant and the refrigerant charge per circuit is 480 kg. [1][2] |
| `q10` | answered | 3318 | The compressor oil must be changed every 4,000 operating hours or 12 months, whichever comes first. [1] |
| `q11` | answered | 3468 | When reinstalling the bearing cap bolts, torque them to **45 N·m** in a star pattern in two passes. [1] |
| `q12` | answered | 3360 | Alarm code E-41 means a condenser water flow loss. To clear the alarm, one needs to check the pump status, strainer, and differential pressure switch. [1] |
| `q13` | answered | 3214 | A maintenance alarm triggers at a condenser approach temperature above 4.2 K. [1] |
| `q14` | answered | 3514 | The CX-900 achieves 0.52 kW/ton at full load under AHRI 550/590 conditions. [1] |
| `q15` | answered | 3775 | The consolidated financial statements for fiscal 2024 were audited by **Harlow & Vance LLP**. The audit opinion issued was unqualified: the financial statements |
| `q16` | answered | 3488 | Management provided revenue guidance of $5,050 million to $5,200 million for fiscal year 2025 [1]. |
| `q17` | abstained | 3560 | I don't know. The provided documents do not contain enough information to answer this question. |
| `q18` | abstained | 3530 | I don't know. The provided documents do not contain enough information to answer this question. |
| `q19` | abstained | 610 | I don't know. The provided documents do not contain enough information to answer this question. |
| `q20` | abstained | 727 | I don't know. The provided documents do not contain enough information to answer this question. |
