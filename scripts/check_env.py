#!/usr/bin/env python
"""Pre-flight check: what works here, and what you still need to configure.

    python scripts/check_env.py

Never prints secret values - only whether they are present.
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.config import get_settings  # noqa: E402
from ragkit.ingest import load_manifest  # noqa: E402
from ragkit.llm import get_llm  # noqa: E402
from ragkit.security import safe_avatar_url  # noqa: E402

OK = "  ok  "
WARN = " warn "
FAIL = " fail "


def check_import(module: str, label: str | None = None) -> bool:
    try:
        importlib.import_module(module)
        print(f"[{OK}] {label or module}")
        return True
    except Exception as exc:
        print(f"[{FAIL}] {label or module} ({type(exc).__name__})")
        return False


def main() -> int:
    settings = get_settings()
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    if sys.version_info < (3, 12):
        print(f"[{FAIL}] Python 3.12+ is required")
        return 1
    print()

    print("Dependencies")
    for module, label in (
        ("streamlit", "streamlit (UI)"),
        ("chromadb", "chromadb (vector store)"),
        ("sentence_transformers", "sentence-transformers (embeddings + reranker)"),
        ("rank_bm25", "rank-bm25 (keyword search)"),
        ("pypdf", "pypdf (PDF)"),
        ("docx", "python-docx (DOCX)"),
        ("ragas", "ragas (evaluation)"),
        ("langchain_openai", "langchain-openai (LLM bridge)"),
        ("supabase", "supabase (auth)"),
        ("jwt", "PyJWT (token verification)"),
    ):
        check_import(module, label)
    print()

    print("Index")
    manifest = load_manifest(settings)
    documents = manifest.get("documents", {})
    if manifest.get("chunk_count"):
        print(f"[{OK}] {manifest['chunk_count']} chunks / {len(documents)} documents")
        print(f"       embedding model used at build time: {manifest.get('embed_model')}")
        if manifest.get("embed_model") != settings.embed_model:
            print(f"[{WARN}] configured embed model differs from index - re-run `make ingest --rebuild`")
    else:
        print(f"[{WARN}] no index yet - run `make ingest`")
    print()

    print("LLM")
    provider = get_llm(settings)
    ok, message = provider.health()
    print(f"[{OK if ok else WARN}] {message}")
    if not ok and settings.llm_provider == "ollama":
        print("       install: curl -fsSL https://ollama.com/install.sh | sh")
        print(f"       then:   ollama pull {settings.ollama_model}")
    if not ok and settings.llm_provider == "openai_compat" and not settings.openai_api_key:
        print("       set RAG_OPENAI_API_KEY (free key: https://console.groq.com/keys)")
        print(f"       current model: {settings.openai_model}")
    print()

    print("Auth")
    if settings.auth_enabled:
        print(f"[{OK}] Supabase configured ({settings.supabase_url})")
        print(f"       admin emails: {len(settings.admin_emails)} configured")
        print(f"       demo avatar host check: {safe_avatar_url('https://example.supabase.co/a.png') is not None}")
    else:
        print(f"[{WARN}] Supabase not configured - app runs in local demo mode")
    if not settings.supabase_jwt_secret:
        print(f"[{WARN}] SUPABASE_JWT_SECRET empty - JWT verification will use JWKS (recommended)")
    print()

    print("Storage")
    for label, path in (
        ("chroma", settings.chroma_dir),
        ("bm25", settings.bm25_path),
        ("chunks", settings.chunks_path),
        ("reports", settings.reports_dir),
    ):
        exists = Path(path).exists()
        print(f"[{OK if exists else WARN}] {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
