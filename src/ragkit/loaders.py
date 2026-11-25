"""Document loading for PDF / DOCX / MD / TXT / HTML with strict limits.

Returns page-level :class:`Page` objects so citations can point at a real page
number, which is what makes the UI citations trustworthy.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .logging_utils import get_logger
from .security import ALLOWED_EXTENSIONS, UploadRejected, iter_safe_paths, validate_upload

log = get_logger("loaders")

_WS_RE = re.compile(r"[ \t\u00a0]+")
_BLANKS_RE = re.compile(r"\n{3,}")


@dataclass
class Page:
    doc_id: str
    source: str
    page: int
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Page":
        return cls(
            doc_id=data["doc_id"],
            source=data["source"],
            page=int(data.get("page", 1)),
            text=data.get("text", ""),
            meta=data.get("meta", {}) or {},
        )


def normalize_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2022", "- ").replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = _WS_RE.sub(" ", text)
    text = _BLANKS_RE.sub("\n\n", text)
    return text.strip()


def doc_id_for(source: str) -> str:
    safe = Path(source).name
    return hashlib.sha1(safe.encode("utf-8")).hexdigest()[:12]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_pdf(data: bytes) -> list[tuple[str, int]]:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover
        raise UploadRejected("missing_dependency", "pypdf is not installed.") from exc

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise UploadRejected("encrypted_pdf", "Password-protected PDFs are not supported.") from exc
    pages: list[tuple[str, int]] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append((normalize_text(text), index))
    if not pages:
        raise UploadRejected("no_text", "No extractable text (is this a scanned PDF?).")
    return pages


def _load_docx(data: bytes) -> list[tuple[str, int]]:
    try:
        import docx
    except Exception as exc:  # pragma: no cover
        raise UploadRejected("missing_dependency", "python-docx is not installed.") from exc

    document = docx.Document(io.BytesIO(data))
    blocks: list[str] = []
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            blocks.append(paragraph.text)
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))
    text = normalize_text("\n\n".join(blocks))
    if not text:
        raise UploadRejected("no_text", "The .docx contains no readable text.")
    return [(text, 1)]


def _load_html(data: bytes) -> list[tuple[str, int]]:
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:  # pragma: no cover
        raise UploadRejected("missing_dependency", "beautifulsoup4 is not installed.") from exc

    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "object"]):
        tag.decompose()
    text = normalize_text(soup.get_text("\n"))
    if not text:
        raise UploadRejected("no_text", "No readable text in the HTML file.")
    return [(text, 1)]


def _load_text(data: bytes) -> list[tuple[str, int]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("cp1252", errors="replace")
    text = normalize_text(text)
    if not text:
        raise UploadRejected("no_text", "The file is empty after normalisation.")
    return [(text, 1)]


def load_bytes(filename: str, data: bytes, max_bytes: int) -> list[Page]:
    """Validate + extract an in-memory upload (used by the Streamlit uploader)."""
    safe_name, digest = validate_upload(filename, data, max_bytes)
    ext = Path(safe_name).suffix.lower()
    if ext == ".pdf":
        raw_pages = _load_pdf(data)
    elif ext == ".docx":
        raw_pages = _load_docx(data)
    elif ext in {".html", ".htm"}:
        raw_pages = _load_html(data)
    elif ext in ALLOWED_EXTENSIONS:
        raw_pages = _load_text(data)
    else:  # pragma: no cover - guarded by validate_upload
        raise UploadRejected("unsupported_type", ext)

    doc_id = digest[:12]
    pages = [
        Page(doc_id=doc_id, source=safe_name, page=page_no, text=text, meta={"sha256": digest})
        for text, page_no in raw_pages
    ]
    log.info("loaded %s (%d page(s), %d chars)", safe_name, len(pages), sum(len(p.text) for p in pages))
    return pages


def load_path(path: Path, max_bytes: int) -> list[Page]:
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise UploadRejected("too_large", f"{path.name} exceeds the per-file limit.")
    return load_bytes(path.name, path.read_bytes(), max_bytes)


def iter_corpus(root: Path, max_bytes: int) -> list[Page]:
    """Load every supported file under ``root``, skipping oversized/unsupported ones."""
    pages: list[Page] = []
    for path in iter_safe_paths(root):
        try:
            pages.extend(load_path(path, max_bytes))
        except UploadRejected as exc:
            log.warning("skipping %s: %s (%s)", path.name, exc.reason, exc.detail)
        except Exception as exc:  # malformed PDF etc. must not kill ingestion
            log.warning("failed to parse %s: %s", path.name, type(exc).__name__)
    return pages


def extract_headings(text: str) -> list[str]:
    """Markdown headings in order of appearance (used for metadata filters)."""
    return [m.group(2).strip() for m in re.finditer(r"(?m)^(#{1,4})\s+(.+)$", text)]
