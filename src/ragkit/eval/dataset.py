"""Golden-set loading and validation.

A 20-question set is small but it is the *right* 20 questions: it covers
single-fact extraction, multi-hop synthesis, numeric precision, tables, refusal
behaviour and prompt-injection resistance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

REQUIRED_FIELDS = ("id", "question", "ground_truth", "answerable")


@dataclass
class GoldenSample:
    id: str
    question: str
    ground_truth: str
    answerable: bool = True
    reference_contexts: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    difficulty: str = "medium"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoldenSample":
        missing = [f for f in REQUIRED_FIELDS if f not in data]
        if missing:
            raise ValueError(f"sample {data.get('id', '?')} missing fields: {missing}")
        return cls(
            id=str(data["id"]),
            question=str(data["question"]).strip(),
            ground_truth=str(data["ground_truth"]).strip(),
            answerable=bool(data.get("answerable", True)),
            reference_contexts=[str(c) for c in data.get("reference_contexts", [])],
            tags=[str(t) for t in data.get("tags", [])],
            difficulty=str(data.get("difficulty", "medium")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_golden_set(path: Path) -> list[GoldenSample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Golden set not found: {path}")
    samples: list[GoldenSample] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_no} is not valid JSON ({exc.msg})") from exc
            sample = GoldenSample.from_dict(data)
            if sample.id in seen:
                raise ValueError(f"duplicate sample id: {sample.id}")
            seen.add(sample.id)
            samples.append(sample)
    if not samples:
        raise ValueError("Golden set is empty.")
    return samples


def validate_golden_set(samples: Iterable[GoldenSample]) -> list[str]:
    problems: list[str] = []
    samples = list(samples)
    if len(samples) < 20:
        problems.append(f"expected at least 20 samples, found {len(samples)}")
    for sample in samples:
        if len(sample.question) < 8:
            problems.append(f"{sample.id}: question is too short")
        if sample.answerable and not sample.reference_contexts:
            problems.append(f"{sample.id}: answerable sample needs reference_contexts")
        if not sample.answerable and "don't know" not in sample.ground_truth.lower() and "do not know" not in sample.ground_truth.lower():
            problems.append(f"{sample.id}: unanswerable samples should have an 'I don't know' ground truth")
    if not any(not s.answerable for s in samples):
        problems.append("include at least one unanswerable control question")
    return problems


def split_by_answerable(samples: Iterable[GoldenSample]) -> tuple[list[GoldenSample], list[GoldenSample]]:
    answerable, unanswerable = [], []
    for sample in samples:
        (answerable if sample.answerable else unanswerable).append(sample)
    return answerable, unanswerable
