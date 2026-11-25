#!/usr/bin/env python
"""CLI: ingest a corpus into Chroma + BM25.

Examples
--------
    python scripts/ingest.py                       # ingest data/corpus
    python scripts/ingest.py --path data/corpus --rebuild
    python scripts/ingest.py --file ~/reports/10-K.pdf
    python scripts/ingest.py --stats               # inspect the index
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.config import get_settings  # noqa: E402
from ragkit.ingest import Ingestor, load_manifest  # noqa: E402
from ragkit.loaders import iter_corpus  # noqa: E402
from ragkit.logging_utils import configure_logging, get_logger  # noqa: E402
from ragkit.rag import RAGPipeline  # noqa: E402

log = get_logger("cli.ingest")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the hybrid index.")
    parser.add_argument("--path", default=None, help="Directory to ingest (default: data/corpus)")
    parser.add_argument("--file", action="append", default=[], help="Specific file(s) to ingest")
    parser.add_argument("--rebuild", action="store_true", help="Drop the existing index first")
    parser.add_argument("--stats", action="store_true", help="Print index statistics and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging("DEBUG" if args.verbose else "INFO")
    settings = get_settings()
    manifest = load_manifest(settings)

    if args.stats:
        print(json.dumps(manifest, indent=2)[:4000] or "No index found. Run make ingest.")
        return 0

    pipeline = RAGPipeline(settings)
    if args.rebuild:
        log.info("resetting index")
        Ingestor(settings).reset()

    if args.file:
        report = pipeline.ingestor().ingest_paths([Path(f) for f in args.file])
    else:
        directory = Path(args.path) if args.path else settings.corpus_dir
        if not directory.exists():
            log.error("corpus directory not found: %s", directory)
            return 2
        pages = iter_corpus(directory, settings.max_file_bytes)
        log.info("loaded %d page(s) from %s", len(pages), directory)
        report = pipeline.ingestor().ingest_pages(pages)

    print(json.dumps(report.to_dict(), indent=2))
    if report.rejected:
        for item in report.rejected:
            log.error("rejected %s: %s (%s)", item["file"], item["reason"], item.get("detail", ""))
        return 1
    if report.chunks == 0:
        log.error("No chunks produced. Is the corpus empty or unreadable?")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
