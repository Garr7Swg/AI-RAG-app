from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from ragkit.config import Settings  # noqa: E402


@pytest.fixture()
def tmp_settings(tmp_path: Path) -> Settings:
    """Settings pointed at a throwaway directory (no model loading is triggered)."""
    settings = Settings(
        project_root=tmp_path,
        corpus_dir=tmp_path / "corpus",
        storage_dir=tmp_path / "storage",
        eval_dir=tmp_path / "eval",
        samples_dir=ROOT / "data" / "eval",
    )
    settings.ensure_dirs()
    return settings
