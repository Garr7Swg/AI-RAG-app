"""Central configuration.

Everything is environment driven so the same code runs in Codespaces, on a
Streamlit Community Cloud app, and in GitHub Actions. No pydantic, no surprises.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """Best-effort .env loading (never overrides real environment variables)."""
    try:
        from dotenv import load_dotenv
    except Exception:  # pragma: no cover - dotenv is optional for tests
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _int(name: str, default: int, lo: int | None = None, hi: int | None = None) -> int:
    try:
        value = int(float(_env(name, str(default))))
    except ValueError:
        value = default
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def _float(name: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    try:
        value = float(_env(name, str(default)))
    except ValueError:
        value = default
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def _csv(name: str) -> tuple[str, ...]:
    raw = _env(name)
    return tuple(part.strip().lower() for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings."""

    # paths
    project_root: Path = PROJECT_ROOT
    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    storage_dir: Path = PROJECT_ROOT / "storage"
    eval_dir: Path = PROJECT_ROOT / "eval"
    samples_dir: Path = PROJECT_ROOT / "data" / "eval"

    # chunking
    chunk_max_chars: int = 1400
    chunk_min_chars: int = 240
    semantic_percentile: float = 88.0
    max_file_mb: float = 10.0

    # retrieval
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_dense: int = 20
    top_k_bm25: int = 20
    top_k_final: int = 5
    rerank_pool: int = 25
    rerank_min_score: float = 0.20
    vector_backend: str = "chroma"
    collection_name: str = "enterprise_docs"

    # generation
    llm_provider: str = "ollama"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b-instruct"
    openai_base_url: str = "https://api.groq.com/openai/v1"
    openai_model: str = "openai/gpt-oss-20b"
    openai_api_key: str = ""
    temperature: float = 0.0
    max_answer_tokens: int = 512
    llm_timeout_s: float = 120.0

    # guardrails
    grounding_min_overlap: float = 0.30
    strict_numeric_grounding: bool = True
    max_question_chars: int = 1200
    rate_limit_per_min: int = 20

    # auth
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_jwt_secret: str = ""
    # Public origin of the running app (Streamlit Cloud, Codespaces port forward,
    # tunnel). Empty means "not configured": Supabase then falls back to the
    # project's Site URL, which is usually localhost and breaks on Cloud.
    public_url: str = ""
    admin_emails: tuple[str, ...] = field(default_factory=tuple)
    session_timeout_min: int = 120

    # optional pinecone
    pinecone_api_key: str = ""
    pinecone_index: str = "rag-enterprise"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # ---------------- derived paths ----------------
    @property
    def chroma_dir(self) -> Path:
        return self.storage_dir / "chroma"

    @property
    def bm25_path(self) -> Path:
        return self.storage_dir / "bm25" / "index.pkl"

    @property
    def chunks_path(self) -> Path:
        return self.storage_dir / "chunks.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.storage_dir / "manifest.json"

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def audit_dir(self) -> Path:
        return self.storage_dir / "audit"

    @property
    def reports_dir(self) -> Path:
        return self.eval_dir / "reports"

    @property
    def history_dir(self) -> Path:
        return self.reports_dir / "history"

    @property
    def golden_set_path(self) -> Path:
        return self.samples_dir / "golden_set.jsonl"

    @property
    def max_file_bytes(self) -> int:
        return int(self.max_file_mb * 1024 * 1024)

    @property
    def auth_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_anon_key)

    @property
    def auth_redirect_url(self) -> str:
        """Where Supabase sends confirmation links.

        ``RAG_PUBLIC_URL`` wins. When it is empty we send nothing, and Supabase
        uses the project's *Site URL* - correct for localhost, wrong on Cloud.
        """
        return self.public_url

    @property
    def auth_redirect_configured(self) -> bool:
        return bool(self.auth_redirect_url)

    def ensure_dirs(self) -> None:
        for directory in (
            self.storage_dir,
            self.chroma_dir,
            self.uploads_dir,
            self.audit_dir,
            self.reports_dir,
            self.history_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def public_view(self) -> dict[str, Any]:
        """Settings safe to render in the UI (no secrets)."""
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.ollama_model if self.llm_provider == "ollama" else self.openai_model,
            "embed_model": self.embed_model,
            "rerank_model": self.rerank_model,
            "vector_backend": self.vector_backend,
            "top_k_final": self.top_k_final,
            "chunk_max_chars": self.chunk_max_chars,
            "max_file_mb": self.max_file_mb,
            "auth_enabled": self.auth_enabled,
            "auth_redirect_url": self.auth_redirect_url or "(Supabase Site URL)",
        }

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        root = Path(_env("RAG_PROJECT_ROOT", str(PROJECT_ROOT))).resolve()

        def p(name: str, default: str) -> Path:
            path = Path(_env(name, default))
            return path if path.is_absolute() else root / path

        openai_api_key = _env("RAG_OPENAI_API_KEY")
        provider = _env("RAG_LLM_PROVIDER", "ollama").lower() or "ollama"
        if provider == "auto":
            # Opt-in shortcut: Groq when a key is configured, Ollama otherwise.
            provider = "openai_compat" if openai_api_key else "ollama"

        return cls(
            project_root=root,
            corpus_dir=p("RAG_CORPUS_DIR", "data/corpus"),
            storage_dir=p("RAG_STORAGE_DIR", "storage"),
            eval_dir=p("RAG_EVAL_DIR", "eval"),
            samples_dir=p("RAG_SAMPLES_DIR", "data/eval"),
            chunk_max_chars=_int("RAG_CHUNK_MAX_CHARS", 1400, 300, 6000),
            chunk_min_chars=_int("RAG_CHUNK_MIN_CHARS", 240, 80, 2000),
            semantic_percentile=_float("RAG_SEMANTIC_PERCENTILE", 88.0, 50.0, 99.0),
            max_file_mb=_float("RAG_MAX_FILE_MB", 10.0, 0.1, 100.0),
            embed_model=_env("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
            rerank_model=_env("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            top_k_dense=_int("RAG_TOP_K_DENSE", 20, 1, 200),
            top_k_bm25=_int("RAG_TOP_K_BM25", 20, 1, 200),
            top_k_final=_int("RAG_TOP_K_FINAL", 5, 1, 20),
            rerank_pool=_int("RAG_RERANK_POOL", 25, 1, 100),
            rerank_min_score=_float("RAG_RERANK_MIN_SCORE", 0.20, 0.0, 0.99),
            vector_backend=_env("RAG_VECTOR_BACKEND", "chroma").lower(),
            collection_name=_env("RAG_COLLECTION", "enterprise_docs"),
            llm_provider=provider,
            ollama_url=_env("RAG_OLLAMA_URL", "http://localhost:11434").rstrip("/"),
            ollama_model=_env("RAG_OLLAMA_MODEL", "qwen2.5:3b-instruct"),
            openai_base_url=_env("RAG_OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
            openai_model=_env("RAG_OPENAI_MODEL", "openai/gpt-oss-20b"),
            openai_api_key=openai_api_key,
            temperature=_float("RAG_TEMPERATURE", 0.0, 0.0, 1.5),
            max_answer_tokens=_int("RAG_MAX_ANSWER_TOKENS", 512, 64, 4096),
            llm_timeout_s=_float("RAG_LLM_TIMEOUT_S", 120.0, 5.0, 600.0),
            grounding_min_overlap=_float("RAG_GROUNDING_MIN_OVERLAP", 0.30, 0.0, 0.95),
            strict_numeric_grounding=_env("RAG_STRICT_NUMERIC", "true").lower() not in {"0", "false", "no"},
            max_question_chars=_int("RAG_MAX_QUESTION_CHARS", 1200, 50, 8000),
            rate_limit_per_min=_int("RAG_RATE_LIMIT_PER_MIN", 20, 1, 600),
            supabase_url=_env("SUPABASE_URL").rstrip("/"),
            supabase_anon_key=_env("SUPABASE_ANON_KEY"),
            supabase_jwt_secret=_env("SUPABASE_JWT_SECRET"),
            public_url=_env("RAG_PUBLIC_URL").rstrip("/"),
            admin_emails=_csv("RAG_ADMIN_EMAILS"),
            session_timeout_min=_int("RAG_SESSION_TIMEOUT_MIN", 120, 5, 10080),
            pinecone_api_key=_env("PINECONE_API_KEY"),
            pinecone_index=_env("PINECONE_INDEX", "rag-enterprise"),
            pinecone_cloud=_env("PINECONE_CLOUD", "aws"),
            pinecone_region=_env("PINECONE_REGION", "us-east-1"),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings.from_env()
    settings.ensure_dirs()
    return settings


def reset_settings_cache() -> None:
    """Used by tests when environment variables change."""
    get_settings.cache_clear()
