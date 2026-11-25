"""Structured logging + tamper-evident-ish audit trail with redaction.

Every request writes one JSON line. Question text is hashed by default (never
stored verbatim) so the log is safe to attach to a GitHub Actions artefact.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CONFIGURED = False
_LOCK = threading.Lock()

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_BEARER_RE = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._\-]{12,}")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b")
_KEY_RE = re.compile(r"(?i)\b(sk-[A-Za-z0-9]{12,}|sbp_[A-Za-z0-9]{12,}|ghp_[A-Za-z0-9]{20,})")


def redact(text: str, max_len: int = 500) -> str:
    """Remove secrets/PII from anything that might reach a log or a UI banner."""
    if not text:
        return ""
    out = _EMAIL_RE.sub("[email]", text)
    out = _BEARER_RE.sub(r"\1[redacted]", out)
    out = _JWT_RE.sub("[jwt]", out)
    out = _KEY_RE.sub("[api-key]", out)
    out = " ".join(out.split())
    return out[:max_len]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def configure_logging(level: str | None = None) -> None:
    global _CONFIGURED
    with _LOCK:
        if _CONFIGURED:
            return
        lvl = (level or os.getenv("RAG_LOG_LEVEL", "INFO")).upper()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s :: %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("ragkit")
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(getattr(logging, lvl, logging.INFO))
        root.propagate = False
        _CONFIGURED = True


def get_logger(name: str = "ragkit") -> logging.Logger:
    configure_logging()
    return logging.getLogger(name if name.startswith("ragkit") else f"ragkit.{name}")


class AuditLog:
    """Append-only JSONL audit log (one file per UTC day)."""

    def __init__(self, directory: Path, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = enabled
        self._lock = threading.Lock()

    def _path(self) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.directory / f"audit-{day}.jsonl"

    def write(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            **{k: redact(v) if isinstance(v, str) else v for k, v in fields.items()},
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                with self._path().open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:  # never let logging break a request
                get_logger().warning("audit write failed: %s", event)
