"""Northwind RAG console - Streamlit front end.

Run locally:  make serve      (http://localhost:8501)
Deploy:       Streamlit Community Cloud, main file = app/streamlit_app.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import theme  # noqa: E402
from ragkit.auth import AuthError, AuthService, Session, User  # noqa: E402
from ragkit.config import get_settings, reset_settings_cache  # noqa: E402
from ragkit.eval.dataset import load_golden_set  # noqa: E402
from ragkit.eval.metrics import EvalUnavailable, heuristic_scores, ragas_scores  # noqa: E402
from ragkit.eval.report import evaluate_gate, load_baseline, to_markdown, write_report  # noqa: E402
from ragkit.eval.report import build_report  # noqa: E402
from ragkit.eval.runner import run_samples  # noqa: E402
from ragkit.ingest import load_manifest  # noqa: E402
from ragkit.llm import LLMError  # noqa: E402
from ragkit.logging_utils import AuditLog, get_logger, redact  # noqa: E402
from ragkit.rag import RAGPipeline  # noqa: E402
from ragkit.security import TokenBucket, rate_limiter  # noqa: E402

log = get_logger("app")

st.set_page_config(
    page_title="Northwind RAG · Enterprise Document Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_UPLOAD_FILES = 8
ALLOWED_UPLOAD_TYPES = ["pdf", "md", "markdown", "txt", "docx", "html", "htm"]
DISCLAIMER = (
    "Free-tier hardware and free API tiers can cold-start in <b>30-60 seconds</b>. "
    "Answers are grounded only in your uploaded documents; when the evidence is weak the "
    "assistant replies <b>&ldquo;I don't know.&rdquo;</b> instead of guessing."
)


# --------------------------------------------------------------------- secrets
def hydrate_secrets() -> None:
    """Bridge Streamlit Cloud secrets into the environment (our config reads env)."""
    try:
        secrets = st.secrets
    except Exception:
        return
    try:
        items = dict(secrets.items())
    except Exception:
        return
    changed = False
    for key, value in items.items():
        if isinstance(value, str) and key not in os.environ:
            os.environ[key] = value
            changed = True
    if changed:
        reset_settings_cache()


hydrate_secrets()
settings = get_settings()


# ------------------------------------------------------------------- resources
@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    pipeline = RAGPipeline(settings)
    try:
        pipeline.warm()
    except Exception as exc:  # model download issues must not kill the UI
        log.warning("warm-up failed: %s", type(exc).__name__)
    return pipeline


@st.cache_resource(show_spinner=False)
def get_auth_service() -> AuthService:
    return AuthService(settings)


@st.cache_resource(show_spinner=False)
def get_limiter(per_minute: int) -> TokenBucket:
    return rate_limiter(per_minute)


def audit() -> AuditLog:
    return AuditLog(settings.audit_dir, enabled=True)


def safe_markdown(text: str) -> str:
    """Strip HTML/JS from model output before rendering as markdown."""
    cleaned = re.sub(r"<[^>]{0,200}>", "", text or "")
    cleaned = re.sub(r"(?i)javascript\s*:", "", cleaned)
    cleaned = re.sub(r"(?i)data:text/html[^)\s]*", "", cleaned)
    return cleaned.strip()


def init_state() -> None:
    defaults = {
        "user": None,
        "session": None,
        "messages": [],
        "history": [],
        "last_citations": [],
        "pending_question": None,
        "booted": False,
        "auth_tab": "sign_in",
        "last_seen": time.time(),
        "page": "Chat",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()
theme.inject_css()


# ------------------------------------------------------------------------ auth
def render_auth() -> None:
    theme.brand_block()
    st.markdown(
        """
        <div class="auth-card">
          <div class="brand-sub">Enterprise document intelligence</div>
          <div class="auth-title">Sign in to Northwind</div>
          <div class="auth-sub">Grounded answers with citations, hybrid retrieval and an automated evaluation gate.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tab_signin, tab_signup = st.tabs(["Sign in", "Create account"])

    auth_service = get_auth_service()

    with tab_signin:
        with st.form("sign_in_form"):
            email = st.text_input("Work email", placeholder="you@company.com", autocomplete="email")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Sign in", use_container_width=True, type="primary")
        if submitted:
            try:
                user, session = auth_service.sign_in(email, password)
                st.session_state.user = user
                st.session_state.session = session
                st.session_state.last_seen = time.time()
                audit().write("auth.sign_in", user_id=user.id[:12], role=user.role)
                st.success("Signed in.")
                st.rerun()
            except AuthError as exc:
                audit().write("auth.sign_in_failed", email=redact(email, 80))
                st.error(str(exc))
            except Exception:
                st.error("Sign-in failed. Please try again.")

    with tab_signup:
        with st.form("sign_up_form"):
            full_name = st.text_input("Full name", placeholder="Ada Lovelace", autocomplete="name")
            new_email = st.text_input("Work email", placeholder="you@company.com", key="su_email")
            new_password = st.text_input(
                "Password", type="password", key="su_password", autocomplete="new-password",
                help="At least 10 characters. A passphrase of 4+ words is ideal.",
            )
            confirm = st.text_input("Confirm password", type="password", key="su_confirm")
            agreed = st.checkbox("I understand this is a demo running on free-tier infrastructure.")
            submitted_up = st.form_submit_button("Create account", use_container_width=True)
        if submitted_up:
            if new_password != confirm:
                st.error("Passwords do not match.")
            elif not agreed:
                st.warning("Please acknowledge the infrastructure notice.")
            else:
                try:
                    needs_confirm, message = auth_service.sign_up(
                        new_email, new_password, full_name,
                        redirect_to=settings.auth_redirect_url,
                    )
                    audit().write("auth.sign_up", email=redact(new_email, 80))
                    (st.info if needs_confirm else st.success)(message)
                except AuthError as exc:
                    st.error(str(exc))
                except Exception:
                    st.error("Could not create the account. Please try again.")

    redirect_target = settings.auth_redirect_configured
    st.caption(
        f"Confirmation links return to `{settings.auth_redirect_url}`."
        if redirect_target
        else "Confirmation links use your Supabase **Site URL**. On a hosted deployment set "
        "`RAG_PUBLIC_URL` to this app's public origin, or the link will point at localhost."
    )

    st.markdown(
        '<div class="fineprint" style="margin-top:1rem">'
        "Authentication is handled by your own Supabase project. This app only ever receives the "
        "access token, verifies it locally (signature + audience + issuer + expiry) and stores it in "
        "the server-side session. It never writes tokens to a file, a URL or a JavaScript-readable cookie."
        "</div>",
        unsafe_allow_html=True,
    )


def require_user() -> User:
    if not settings.auth_enabled:
        if st.session_state.user is None:
            st.session_state.user = get_auth_service().demo_user()
        return st.session_state.user

    user: User | None = st.session_state.user
    session: Session | None = st.session_state.session
    if user is None or session is None:
        render_auth()
        st.stop()

    # idle timeout
    idle_minutes = (time.time() - float(st.session_state.get("last_seen", time.time()))) / 60
    if idle_minutes > settings.session_timeout_min:
        st.session_state.user = None
        st.session_state.session = None
        st.warning("Your session expired after a period of inactivity. Please sign in again.")
        render_auth()
        st.stop()

    try:
        user = get_auth_service().ensure_valid(user, session)
        st.session_state.user = user
    except AuthError as exc:
        st.session_state.user = None
        st.session_state.session = None
        st.warning(str(exc))
        render_auth()
        st.stop()

    st.session_state.last_seen = time.time()
    return user


user = require_user()
if st.session_state.get("booted") is False:
    slot = st.empty()
    theme.splash(slot, "Loading models and index…")
    pipeline = get_pipeline()
    slot.empty()
    st.session_state.booted = True
else:
    pipeline = get_pipeline()

manifest = load_manifest(settings)


# --------------------------------------------------------------------- sidebar
def _short(label: str, limit: int = 26) -> str:
    return label if len(label) <= limit else label[: limit - 1] + "…"


with st.sidebar:
    theme.brand_block()

    if st.button("＋  New chat", key="new_chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_citations = []
        st.session_state.nav = "Chat"
        st.rerun()

    page = st.radio(
        "View",
        ["Chat", "Library", "Evaluation", "System"],
        label_visibility="collapsed",
        key="nav",
    )

    chunk_count = manifest.get("chunk_count", 0)
    doc_count = len(manifest.get("documents", {}))
    documents_in_index = manifest.get("documents", {})

    theme.sidebar_label("Collections")
    if documents_in_index:
        rows = [(_short(entry.get("source", doc_id)), str(entry.get("chunks", 0)))
                for doc_id, entry in sorted(documents_in_index.items())]
        theme.collection_list(rows[:6], active=rows[0][0])
    else:
        theme.collection_list([("No documents indexed", "0")])

    theme.sidebar_label("Recent chats")
    theme.recent_list(list(st.session_state.get("history", []))[-5:])

    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
    theme.user_card(
        user.display_name,
        user.email,
        "admin" if user.is_admin else ("demo" if user.is_demo else "analyst"),
        user.avatar_url,
    )
    if st.button("Sign out", use_container_width=True):
        try:
            get_auth_service().sign_out(st.session_state.get("session"))
        finally:
            audit().write("auth.sign_out", user_id=user.id[:12])
            st.session_state.user = None
            st.session_state.session = None
            st.session_state.messages = []
            st.session_state.last_citations = []
            st.session_state.last_seen = 0.0
            st.rerun()

    llm_label = settings.ollama_model if settings.llm_provider == "ollama" else settings.openai_model
    st.caption(f"{doc_count} files · {chunk_count} chunks · `{settings.embed_model.split('/')[-1]}` · `{llm_label}`")
    if user.is_demo:
        st.caption(":orange[Demo mode — Supabase not configured]")


# ------------------------------------------------------------------------ chat
def render_answer_html(text: str) -> str:
    """Render assistant markdown, turning bounded ``[n]`` markers into citation chips."""
    cleaned = safe_markdown(text)
    return re.sub(
        r"\[(\d{1,2})\]",
        lambda match: f'<span class="cite c{(int(match.group(1)) - 1) % 4 + 1}">{match.group(1)}</span>',
        cleaned,
    )


def render_message(message: dict) -> None:
    role = message.get("role", "user")
    with st.chat_message(role):
        if role == "assistant":
            st.markdown(render_answer_html(message.get("content", "")), unsafe_allow_html=True)
        else:
            st.markdown(safe_markdown(message.get("content", "")))
        meta = message.get("meta")
        if meta:
            st.markdown(f'<div class="answer-meta">{meta}</div>', unsafe_allow_html=True)


SUGGESTIONS = (
    "Revenue in fiscal 2024?",
    "Cross-default threshold?",
    "CX-900 maintenance?",
)


def chat_page() -> None:
    chunk_count = manifest.get("chunk_count", 0)
    doc_count = len(manifest.get("documents", {}))
    theme.topbar("Product docs", f"{chunk_count} chunks indexed · {doc_count} files")

    chat_col, sources_col = st.columns([0.635, 0.365], gap="large")

    with sources_col:
        last_citations = st.session_state.get("last_citations") or []
        theme.sources_header(len(last_citations))
        if last_citations:
            theme.source_cards(last_citations)
        else:
            theme.sources_empty()
        theme.rerank_card(settings.rerank_min_score, settings.top_k_final)

    with chat_col:
        pending = st.session_state.pop("pending_question", None)

        if not chunk_count:
            theme.notice(
                "No documents indexed yet. Open <b>Library</b> to upload, or run <code>make ingest</code> "
                "in the terminal.",
                "info",
                "ℹ️",
            )

        if not st.session_state.messages and not pending:
            theme.greeting(
                "Ask your documents",
                "Hybrid retrieval, cross-encoder re-ranking and citation checks on every answer. "
                "Answers cite the passages shown on the right.",
            )
            chip_cols = st.columns(3)
            for index, prompt in enumerate(SUGGESTIONS, start=1):
                if chip_cols[index - 1].button(prompt, key=f"sugg_{index}", use_container_width=True):
                    st.session_state.pending_question = prompt
                    st.rerun()

        for message in st.session_state.messages:
            render_message(message)

        question = st.chat_input("Ask anything about your documents")
        question = question or pending
        if not question:
            st.markdown(
                '<div class="fineprint" style="margin:.4rem 0 .2rem .5rem">'
                "Free-tier hardware can take 30-60 s to warm up. When the evidence is weak the "
                "assistant answers &ldquo;I don't know.&rdquo; instead of guessing.</div>",
                unsafe_allow_html=True,
            )
            return

        decision = get_limiter(settings.rate_limit_per_min).check(user.id)
        if not decision.allowed:
            st.warning(f"Rate limit reached. Try again in {decision.retry_after_s:.0f}s.")
            return

        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.history = (list(st.session_state.get("history", [])) + [question])[-20:]
        with st.chat_message("user"):
            st.markdown(safe_markdown(question))

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-7:-1]
        if m.get("role") in {"user", "assistant"}
    ]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Searching documents and checking evidence…_")
        started = time.perf_counter()
        final = None
        streamed = ""
        try:
            for kind, payload in pipeline.stream_answer(question, history=history):
                if kind == "token":
                    streamed += payload
                    placeholder.markdown(render_answer_html(streamed) + "▌", unsafe_allow_html=True)
                elif kind == "result":
                    final = payload
        except LLMError as exc:
            placeholder.error(redact(str(exc), 240))
            st.session_state.messages.append(
                {"role": "assistant", "content": "I don't know. The language model is unavailable right now."}
            )
            return
        except Exception as exc:
            log.warning("answer failed: %s", type(exc).__name__)
            placeholder.error("Something went wrong while answering. Please retry.")
            return

        if final is None:
            placeholder.warning("No answer was produced.")
            return

        placeholder.markdown(render_answer_html(final.text), unsafe_allow_html=True)
        citations = [c.to_dict() for c in final.citations]
        st.session_state.last_citations = citations
        elapsed = (time.perf_counter() - started) * 1000
        meta = (
            f"{'Abstained' if final.abstained else 'Answered'} · {elapsed:.0f} ms · "
            f"retrieval {final.latency_ms:.0f} ms · `{final.model}`"
            + (f" · reason `{final.abstain_reason}`" if final.abstained else "")
        )
        st.markdown(f'<div class="answer-meta">{meta}</div>', unsafe_allow_html=True)

        with st.expander("Retrieval trace"):
            for rank, item in enumerate(final.retrieved, start=1):
                st.markdown(
                    f"`{rank}` **{item.get('source', '?')}** p.{item.get('page', '?')} · "
                    f"score `{float(item.get('score', 0)):.3f}` · dense "
                    f"`{item.get('dense_rank')}` · bm25 `{item.get('bm25_rank')}`"
                )
                st.caption(str(item.get("snippet", ""))[:280])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final.text,
            "citations": citations,
            "meta": meta,
        }
    )
    audit().write(
        "rag.answer",
        user_id=user.id[:12],
        abstained=final.abstained,
        reason=final.abstain_reason,
        latency_ms=round(elapsed, 1),
        evidence=len(citations),
    )
    # Re-render once so the Sources panel shows this answer's passages.
    st.rerun()


def transcript_downloads() -> None:
    messages = st.session_state.get("messages", [])
    if not messages:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    markdown_lines = [f"# Northwind RAG transcript\n\n_{datetime.now(timezone.utc).isoformat()}_\n"]
    for message in messages:
        role = "You" if message["role"] == "user" else "Assistant"
        markdown_lines.append(f"\n## {role}\n\n{message.get('content', '')}\n")
        for citation in message.get("citations") or []:
            markdown_lines.append(
                f"- `{citation.get('source')}` p.{citation.get('page')} (score {citation.get('score')})\n"
            )
    col_a, col_b = st.columns(2)
    col_a.download_button(
        "Download transcript (.md)",
        data="\n".join(markdown_lines).encode("utf-8"),
        file_name=f"rag-transcript-{stamp}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    col_b.download_button(
        "Download transcript (.json)",
        data=json.dumps(messages, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name=f"rag-transcript-{stamp}.json",
        mime="application/json",
        use_container_width=True,
    )


# --------------------------------------------------------------------- library
def library_page() -> None:
    theme.hero(
        "Document library",
        "Upload PDF, DOCX, Markdown, TXT or HTML. Files are validated, chunked semantically and indexed locally.",
        pills=[(f"{settings.max_file_mb:.0f} MB limit", theme.ACCENT_2), ("Content-addressed", theme.OK)],
    )

    documents = manifest.get("documents", {})
    if documents:
        rows = [
            {
                "document": entry.get("source", doc_id),
                "doc_id": doc_id,
                "chunks": entry.get("chunks", 0),
                "pages": entry.get("pages", 0),
                "ingested": entry.get("ingested_at", ""),
                "flags": ", ".join(entry.get("injection_flags", [])) or "—",
            }
            for doc_id, entry in sorted(documents.items())
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        theme.notice("The library is empty. Upload a document below or run <code>make ingest</code>.", "info", "ℹ️")

    st.markdown("### Upload documents")
    uploaded = st.file_uploader(
        "Drop files here",
        type=ALLOWED_UPLOAD_TYPES,
        accept_multiple_files=True,
        help=f"Up to {MAX_UPLOAD_FILES} files, {settings.max_file_mb:.0f} MB each.",
    )
    col_ingest, col_rebuild = st.columns([2, 1])
    ingest_clicked = col_ingest.button("Ingest uploaded files", type="primary", use_container_width=True)
    rebuild_clicked = col_rebuild.button("Re-ingest corpus", use_container_width=True)

    if rebuild_clicked:
        if not user.is_admin:
            st.error("Only workspace admins can re-index the corpus.")
        else:
            with st.spinner("Re-ingesting data/corpus…"):
                try:
                    report = pipeline.ingest_corpus(rebuild=True)
                    st.success(f"Indexed {report['chunks']} chunks from {report['documents']} documents.")
                    time.sleep(1)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Re-ingest failed: {redact(str(exc), 160)}")

    if ingest_clicked:
        if not uploaded:
            st.warning("Choose at least one file first.")
        elif len(uploaded) > MAX_UPLOAD_FILES:
            st.error(f"Please upload at most {MAX_UPLOAD_FILES} files at a time.")
        else:
            files = [(item.name, item.getvalue()) for item in uploaded]
            progress = st.progress(0.0, text="Validating uploads…")
            try:
                report = pipeline.ingest_uploads(files)
            except Exception as exc:
                progress.empty()
                st.error(f"Ingestion failed: {redact(str(exc), 160)}")
                return
            progress.progress(1.0, text="Done")
            if report.get("rejected"):
                for item in report["rejected"]:
                    st.error(f"{item['file']}: {item['reason']} — {item.get('detail', '')}")
            if report.get("chunks"):
                st.success(
                    f"Indexed {report['chunks']} chunks from {report['documents']} document(s). "
                    f"Total index: {report['total_chunks']} chunks."
                )
                audit().write(
                    "library.upload",
                    user_id=user.id[:12],
                    files=len(files),
                    chunks=report["chunks"],
                )
                time.sleep(0.6)
                st.rerun()
            else:
                st.warning("No text could be extracted from those files.")

    st.markdown("---")
    st.markdown("### Downloads")
    col1, col2, col3 = st.columns(3)
    if manifest:
        col1.download_button(
            "Index manifest (.json)",
            data=json.dumps(manifest, indent=2).encode("utf-8"),
            file_name="index-manifest.json",
            mime="application/json",
            use_container_width=True,
        )
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["doc_id", "document", "chunks", "pages", "ingested_at"])
    for doc_id, entry in documents.items():
        writer.writerow([doc_id, entry.get("source", ""), entry.get("chunks", 0), entry.get("pages", 0), entry.get("ingested_at", "")])
    col2.download_button(
        "Library index (.csv)",
        data=buffer.getvalue().encode("utf-8"),
        file_name="library-index.csv",
        mime="text/csv",
        use_container_width=True,
    )
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(settings.corpus_dir.glob("*")):
            if path.is_file():
                archive.write(path, arcname=path.name)
    col3.download_button(
        "Demo corpus (.zip)",
        data=bundle.getvalue(),
        file_name="corpus.zip",
        mime="application/zip",
        use_container_width=True,
    )
    transcript_downloads()

    if documents and user.is_admin:
        with st.expander("Delete a document"):
            target = st.selectbox(
                "Document",
                options=list(documents.keys()),
                format_func=lambda d: documents[d].get("source", d),
            )
            confirm = st.checkbox("I want to remove this document and its vectors")
            if st.button("Delete document", disabled=not confirm):
                removed = pipeline.ingestor().delete_document(target)
                st.success(f"Removed {removed} chunks.")
                audit().write("library.delete", user_id=user.id[:12], doc_id=target)
                time.sleep(0.5)
                st.rerun()


# ------------------------------------------------------------------ evaluation
def load_latest_report() -> dict | None:
    path = settings.reports_dir / "latest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def evaluation_page() -> None:
    theme.hero(
        "Evaluation gate",
        "Ragas scores faithfulness, answer relevancy and context recall against a 20-question golden set. "
        "The gate fails the build when quality drops.",
        pills=[("Ragas", theme.ACCENT), ("Deterministic fallback", theme.ACCENT_2), ("CI-gated", theme.OK)],
    )
    report = load_latest_report()

    if report:
        metrics = report.get("metrics", {})
        gate = report.get("gate", {})
        passed = bool(gate.get("passed"))
        theme.metric_cards(
            [
                ("Faithfulness", f"{metrics.get('faithfulness', 0):.3f}", "grounded in evidence"),
                ("Answer relevancy", f"{metrics.get('answer_relevancy', 0):.3f}", "answers the question"),
                ("Context recall", f"{metrics.get('context_recall', 0):.3f}", "evidence retrieved"),
                ("Gate", "PASS" if passed else "FAIL", f"judge: {report.get('judge')}"),
            ]
        )
        if passed:
            st.success(f"Gate passed on {report.get('generated_at')} · commit `{report.get('git_sha')}`")
        else:
            st.error("Gate failed:\n" + "\n".join(f"- {f}" for f in gate.get("failures", [])))
        for note in report.get("notes", []):
            st.caption(f"ℹ️ {note}")

        with st.expander("Per-question results", expanded=False):
            st.dataframe(
                [
                    {
                        "id": record["id"],
                        "question": record["question"][:90],
                        "behaviour": "abstained" if record["abstained"] else "answered",
                        "latency_ms": record["latency_ms"],
                        "answer": (record["answer"] or "")[:120],
                    }
                    for record in report.get("records", [])
                ],
                use_container_width=True,
                hide_index=True,
            )

        col1, col2, col3 = st.columns(3)
        col1.download_button(
            "Report (.json)",
            data=json.dumps(report, indent=2).encode("utf-8"),
            file_name="rag-eval-report.json",
            mime="application/json",
            use_container_width=True,
        )
        col2.download_button(
            "Report (.md)",
            data=to_markdown(report).encode("utf-8"),
            file_name="rag-eval-report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        col3.download_button(
            "History snapshot (.jsonl)",
            data="\n".join(json.dumps(r, ensure_ascii=False) for r in report.get("records", [])).encode("utf-8"),
            file_name="rag-eval-history.jsonl",
            mime="application/x-ndjson",
            use_container_width=True,
        )
    else:
        theme.notice(
            "No evaluation report yet. Run one below, or run <code>make eval-offline</code> / "
            "<code>make eval</code> in the terminal.",
            "info",
            "ℹ️",
        )

    st.markdown("### Run an evaluation")
    st.caption(
        "The offline judge needs no model server and finishes in seconds. The Ragas judge calls the "
        "configured LLM and is much slower on free-tier hardware."
    )
    col_a, col_b = st.columns([1, 1])
    sample_limit = col_a.number_input("Questions to run", min_value=1, max_value=20, value=5, step=1)
    use_ragas = col_b.toggle("Use Ragas LLM judge", value=False)

    if not manifest.get("chunk_count"):
        st.warning("Index a document first — evaluation needs evidence to retrieve.")
        return

    if st.button("Run evaluation", type="primary"):
        progress = st.progress(0.0, text="Starting…")
        try:
            samples = load_golden_set(settings.golden_set_path)[: int(sample_limit)]
        except Exception as exc:
            st.error(f"Golden set unavailable: {exc}")
            return

        def on_progress(done: int, total: int, sample_id: str) -> None:
            progress.progress(done / max(1, total), text=f"[{done}/{total}] {sample_id}")

        with st.spinner("Running the pipeline over the golden set…"):
            records = run_samples(pipeline, samples, mode="llm", progress=on_progress)
            metrics_scores = heuristic_scores(records)
            judge_used = "heuristic"
            notes: list[str] = []
            if use_ragas:
                try:
                    ragas_metrics, meta = ragas_scores(records, settings, embedder=pipeline.embedder)
                    metrics_scores.update(ragas_metrics)
                    judge_used = "ragas"
                    notes.append(f"Ragas scored {meta['samples_scored']} samples with `{meta['judge']}`.")
                except EvalUnavailable as exc:
                    notes.append(f"Ragas unavailable: {redact(str(exc), 160)}. Heuristic judge used instead.")

            gate = evaluate_gate(metrics_scores, judge_used, load_baseline(settings, judge=judge_used))
            report = build_report(records, metrics_scores, judge_used, settings, "llm", gate, notes)
            write_report(report, settings)
            audit().write(
                "eval.run",
                user_id=user.id[:12],
                judge=judge_used,
                samples=len(records),
                gate_passed=gate.passed,
            )
        progress.progress(1.0, text="Complete")
        if gate.passed:
            st.success("Gate passed. Report saved to eval/reports/.")
        else:
            st.error("Gate failed: " + "; ".join(gate.failures))
        time.sleep(0.8)
        st.rerun()


# ---------------------------------------------------------------------- system
def system_page() -> None:
    theme.hero(
        "System health",
        "What is loaded, what is configured, and which guardrails are active.",
        pills=[("JWT verified", theme.OK), ("Uploads validated", theme.OK), ("Prompt fencing", theme.ACCENT_2)],
    )
    health = pipeline.health()
    theme.metric_cards(
        [
            ("Index", "ready" if health["index_ready"] else "empty", health["index_message"]),
            ("Documents", str(health["documents"]), f"{health['chunks']} chunks"),
            ("LLM", "ready" if health["llm_ready"] else "offline", health["llm_message"][:60]),
            ("Auth", "Supabase" if settings.auth_enabled else "demo", "JWT verified server-side"),
        ]
    )
    st.markdown("### Configuration")
    st.json(settings.public_view(), expanded=False)

    st.markdown("### Guardrails")
    st.markdown(
        f"""
| Control | Value |
| --- | --- |
| Max upload size | {settings.max_file_mb:.0f} MB |
| Allowed types | PDF, DOCX, MD, TXT, HTML |
| Prompt-injection scan | every chunk, flagged and fenced |
| Grounding overlap floor | {settings.grounding_min_overlap:.2f} |
| Numeric grounding | {"strict" if settings.strict_numeric_grounding else "off"} |
| Retrieval floor (rerank) | {settings.rerank_min_score:.2f} |
| Rate limit | {settings.rate_limit_per_min} / min / user |
| Session timeout | {settings.session_timeout_min} min |
| JWT checks | signature, `aud`, `iss`, `exp` |
"""
    )
    if not settings.auth_enabled:
        theme.notice(
            "Supabase is not configured, so the app is in local demo mode. Set "
            "<code>SUPABASE_URL</code> and <code>SUPABASE_ANON_KEY</code> to enable real accounts.",
            "info",
            "ℹ️",
        )
    elif not settings.auth_redirect_configured:
        theme.notice(
            "<code>RAG_PUBLIC_URL</code> is not set, so Supabase confirmation emails fall back to "
            "the project's <b>Site URL</b> — usually <code>http://localhost:8501</code>, which is "
            "unreachable from a hosted app. Set it to this app's public origin, e.g. "
            "<code>https://your-app.streamlit.app</code>.",
            "info",
            "ℹ️",
        )
    st.caption("Audit log: storage/audit/ (JSONL, one line per event, PII redacted, questions hashed).")


# ------------------------------------------------------------------------ route
if page == "Chat":
    chat_page()
elif page == "Library":
    library_page()
elif page == "Evaluation":
    evaluation_page()
else:
    system_page()
