"""Defensive security primitives: upload validation, prompt-injection scanning,
rate limiting, secret redaction helpers and safe-rendering utilities.

Design notes
------------
* Allowlist, not blocklist, for file types.
* Magic-byte sniffing so a `.pdf` that is really an .exe is rejected.
* Zip-bomb guard for OOXML (`.docx`) containers.
* Uploads are never written next to code: they land in ``storage/uploads``
  under a content-addressed name, which removes path traversal entirely.
* User-supplied text is treated as data. Untrusted context is fenced with
  explicit delimiters and the system prompt says it must not be obeyed.
"""

from __future__ import annotations

import hashlib
import io
import re
import threading
import time
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

MAX_FILENAME_LEN = 120
ALLOWED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ".text", ".docx", ".html", ".htm"}

# 25 MB uncompressed ceiling for a docx that claims <=10 MB compressed.
MAX_UNCOMPRESSED_BYTES = 25 * 1024 * 1024
MAX_ZIP_RATIO = 120

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._ ()\[\]-]+")

INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("override_instructions", re.compile(r"(?i)\b(ignore|disregard|forget)\b[^.\n]{0,40}\b(previous|prior|above|earlier)\b[^.\n]{0,30}\b(instruction|prompt|rule|direction)s?\b")),
    ("reveal_system_prompt", re.compile(r"(?i)\b(reveal|show|print|repeat|dump)\b[^.\n]{0,30}\b(system prompt|initial prompt|instructions you were given|hidden prompt)\b")),
    ("role_hijack", re.compile(r"(?i)\b(you are now|act as|pretend to be|new role)\b[^.\n]{0,40}\b(admin|developer|dan|unrestricted|root)\b")),
    ("exfiltration", re.compile(r"(?i)\b(send|post|upload|forward)\b[^.\n]{0,40}\b(api key|secret|token|credential|password)\b")),
    ("chat_template_tokens", re.compile(r"<\|(?:im_start|im_end|system|endoftext|assistant)\|>")),
    ("fake_delimiter", re.compile(r"(?i)</?(?:system|context|document|instructions)>")),
    ("data_uri_payload", re.compile(r"(?i)data:text/html;base64,")),
)

_SECRET_KEYWORDS = re.compile(r"(?i)(password|passwd|secret|api[_ -]?key|access[_ -]?token)")

SAFE_AVATAR_HOSTS = (
    "supabase.co",
    "supabase.in",
    "gravatar.com",
    "githubusercontent.com",
    "googleusercontent.com",
    "cloudflare-ipfs.com",
)


class UploadRejected(ValueError):
    """Raised when an uploaded artefact fails validation."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


@dataclass
class InjectionFinding:
    kind: str
    snippet: str


def sanitize_filename(name: str) -> str:
    """Return a safe basename. Raises UploadRejected when nothing is left."""
    original = name or ""
    base = Path(original.replace("\\", "/")).name
    base = unicodedata.normalize("NFKC", base)
    base = _CONTROL_RE.sub("", base)
    base = base.strip()
    if base.startswith("."):
        raise UploadRejected("invalid_filename", "Hidden files are not accepted.")
    base = _SAFE_NAME_RE.sub("_", base).strip(" ._-")
    if not base:
        raise UploadRejected("invalid_filename", "File name is empty after sanitisation.")
    if len(base) > MAX_FILENAME_LEN:
        suffix = Path(base).suffix[:12]
        base = base[: MAX_FILENAME_LEN - len(suffix)] + suffix
    return base


def _looks_like_pdf(head: bytes) -> bool:
    return head.lstrip()[:5] == b"%PDF-"


def _validate_docx(data: bytes) -> None:
    """Guard against zip bombs and non-OOXML zips renamed to .docx."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            if "[Content_Types].xml" not in names:
                raise UploadRejected("bad_docx", "Missing [Content_Types].xml (not a Word file).")
            total = 0
            for info in zf.infolist():
                total += info.file_size
                if total > MAX_UNCOMPRESSED_BYTES:
                    raise UploadRejected("zip_bomb", "Uncompressed size exceeds the 25 MB limit.")
                if info.compress_size > 0 and info.file_size / info.compress_size > MAX_ZIP_RATIO:
                    raise UploadRejected("zip_bomb", "Suspicious compression ratio.")
                if ".." in info.filename or info.filename.startswith("/"):
                    raise UploadRejected("bad_docx", "Archive contains unsafe paths.")
    except zipfile.BadZipFile as exc:
        raise UploadRejected("bad_docx", "File is not a valid .docx archive.") from exc


def _validate_text(data: bytes, ext: str) -> None:
    if b"\x00" in data[:4096]:
        raise UploadRejected("binary_content", f"{ext} upload contains binary data.")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = data.decode("cp1252")
        except UnicodeDecodeError as exc:
            raise UploadRejected("encoding", "Text must be UTF-8 (or Windows-1252).") from exc
    if not text.strip():
        raise UploadRejected("empty", "File is empty.")


def validate_upload(filename: str, data: bytes, max_bytes: int) -> tuple[str, str]:
    """Validate an upload. Returns ``(safe_name, content_sha256)``.

    Raises :class:`UploadRejected` with a machine-readable ``reason``.
    """
    if not data:
        raise UploadRejected("empty", "No bytes received.")
    if len(data) > max_bytes:
        raise UploadRejected(
            "too_large",
            f"File is {len(data) / 1048576:.1f} MB; the limit is {max_bytes / 1048576:.0f} MB.",
        )
    safe_name = sanitize_filename(filename)
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadRejected(
            "unsupported_type",
            f"{ext or 'no extension'} is not allowed. Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    if ext == ".pdf" and not _looks_like_pdf(data[:1024]):
        raise UploadRejected("content_mismatch", "File claims to be PDF but lacks a %PDF header.")
    if ext == ".docx":
        _validate_docx(data)
    elif ext in {".txt", ".text", ".md", ".markdown", ".html", ".htm"}:
        _validate_text(data, ext)
    return safe_name, hashlib.sha256(data).hexdigest()


def store_upload(directory: Path, data: bytes, digest: str, safe_name: str) -> Path:
    """Content-addressed storage: the name is derived from the hash, not the user."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{digest[:16]}{Path(safe_name).suffix.lower()}"
    if not target.exists():
        target.write_bytes(data)
    return target


def scan_for_injection(text: str) -> list[InjectionFinding]:
    """Flag prompt-injection style content. Data is still ingested (with a warning
    banner in prompts) because blocking documents outright hides real content."""
    findings: list[InjectionFinding] = []
    if not text:
        return findings
    sample = text[:200_000]
    for kind, pattern in INJECTION_PATTERNS:
        match = pattern.search(sample)
        if match:
            start = max(0, match.start() - 30)
            findings.append(InjectionFinding(kind, sample[start : match.end() + 30].strip()[:180]))
    return findings


def neutralize_chat_tokens(text: str) -> str:
    """Strip control tokens that some models treat as instructions."""
    out = re.sub(r"<\|[^|]{0,20}\|>", " ", text or "")
    return out.replace("\x00", " ")


def truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def clean_question(question: str, limit: int) -> str:
    q = unicodedata.normalize("NFKC", (question or "")).strip()
    q = _CONTROL_RE.sub(" ", q)
    q = re.sub(r"\s+", " ", q)
    if not q:
        raise ValueError("Question is empty.")
    if len(q) > limit:
        raise ValueError(f"Question is longer than {limit} characters.")
    return q


def safe_avatar_url(url: str | None) -> str | None:
    """Only allow https avatars from known image hosts (prevents HTML/JS injection
    through a `javascript:` or tracking URL)."""
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme != "https" or not parsed.hostname:
        return None
    host = parsed.hostname.lower()
    if not any(host == h or host.endswith("." + h) for h in SAFE_AVATAR_HOSTS):
        return None
    if parsed.path.lower().endswith((".svg", ".html", ".htm", ".js")):
        return None
    return url


def looks_like_secret(text: str) -> bool:
    return bool(_SECRET_KEYWORDS.search(text or ""))


def snippet(text: str, limit: int = 260) -> str:
    flat = " ".join((text or "").split())
    return truncate(flat, limit)


@dataclass
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after_s: float


@dataclass
class TokenBucket:
    """Per-identity token bucket. Cheap, thread safe, no Redis required."""

    capacity: int
    refill_per_second: float
    _state: dict[str, tuple[float, float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def check(self, key: str) -> RateLimitDecision:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._state.get(key, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill_per_second)
            if tokens < 1.0:
                self._state[key] = (tokens, now)
                wait = (1.0 - tokens) / max(self.refill_per_second, 1e-6)
                return RateLimitDecision(False, 0, round(wait, 2))
            tokens -= 1.0
            self._state[key] = (tokens, now)
            return RateLimitDecision(True, int(tokens), 0.0)

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._state.clear()
            else:
                self._state.pop(key, None)


def rate_limiter(per_minute: int) -> TokenBucket:
    return TokenBucket(capacity=max(1, per_minute), refill_per_second=max(1, per_minute) / 60.0)


def iter_safe_paths(root: Path) -> Iterable[Path]:
    """Walk a corpus directory without escaping it (symlinks are skipped)."""
    root = Path(root).resolve()
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if root not in resolved.parents and resolved != root:
            continue
        if path.suffix.lower() in ALLOWED_EXTENSIONS:
            yield path
