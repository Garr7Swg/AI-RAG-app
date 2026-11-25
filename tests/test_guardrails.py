"""The abstention logic is the product's core promise - test it directly."""

from __future__ import annotations

from ragkit.hybrid import Retrieved
from ragkit.prompts import IDK_TEXT
from ragkit.rag import RAGPipeline, content_tokens, numbers_in

CONTEXT = (
    "[1] northwind_annual_report_fy2024.md (page 1) — Financial Highlights\n"
    "Consolidated revenue for fiscal year 2024 was $4,812 million, an increase of 7.4% from "
    "$4,480 million in fiscal year 2023. Adjusted EBITDA was $842 million, equal to 17.5% of revenue."
)

KEPT = [
    Retrieved(
        chunk_id="c1",
        text=CONTEXT,
        metadata={"source": "northwind_annual_report_fy2024.md", "page": 1, "heading": "Financial Highlights"},
        score=0.91,
    )
]


def _pipeline(tmp_settings) -> RAGPipeline:
    return RAGPipeline(tmp_settings)  # models stay lazy; _verify is pure


def test_numbers_in_handles_currency_and_percentages():
    assert numbers_in("$4,812 million, up 7.4%") == ["4,812", "7.4"]


def test_content_tokens_drop_stopwords():
    tokens = content_tokens("The revenue was strong in the quarter")
    assert "revenue" in tokens and "the" not in tokens


def test_grounded_answer_passes(tmp_settings):
    pipeline = _pipeline(tmp_settings)
    answer = "Revenue for fiscal 2024 was $4,812 million, up 7.4% from fiscal 2023 [1]."
    citations = pipeline._citations_from([1], KEPT)
    ok, reason, _ = pipeline._verify(answer, KEPT, citations)
    assert ok, reason


def test_hallucinated_number_is_rejected(tmp_settings):
    pipeline = _pipeline(tmp_settings)
    answer = "Revenue for fiscal 2024 was $6,240 million, up 12.1% from fiscal 2023 [1]."
    citations = pipeline._citations_from([1], KEPT)
    ok, reason, citations = pipeline._verify(answer, KEPT, citations)
    assert not ok
    assert reason.startswith("unsupported_number")
    assert answer != IDK_TEXT


def test_unsupported_claim_is_rejected_by_grounding_floor(tmp_settings):
    pipeline = _pipeline(tmp_settings)
    answer = "The company plans to acquire a semiconductor foundry in Malaysia next quarter [1]."
    citations = pipeline._citations_from([1], KEPT)
    ok, reason, _ = pipeline._verify(answer, KEPT, citations)
    assert not ok
    assert reason.startswith("low_grounding")


def test_answer_without_citations_gets_evidence_attached(tmp_settings):
    pipeline = _pipeline(tmp_settings)
    answer = "Adjusted EBITDA was $842 million, equal to 17.5% of revenue."
    ok, reason, citations = pipeline._verify(answer, KEPT, [])
    assert ok, reason
    assert citations and citations[0].chunk_id == "c1"


def test_empty_answer_is_rejected(tmp_settings):
    pipeline = _pipeline(tmp_settings)
    ok, reason, _ = pipeline._verify("   ", KEPT, [])
    assert not ok and reason == "empty_answer"


def test_out_of_range_citation_is_rejected(tmp_settings):
    pipeline = _pipeline(tmp_settings)
    ok, reason, _ = pipeline._verify("Some claim about revenue [9].", KEPT, [])
    assert not ok
    assert reason in {"citation_out_of_range", "no_citations", "low_grounding"}


def test_refusal_is_recognised(tmp_settings):
    pipeline = _pipeline(tmp_settings)
    ok, reason, citations = pipeline._verify(IDK_TEXT, KEPT, [])
    assert ok and reason == "model_abstained" and citations == []
