"""Extract and chunk every downloaded circular into data/processed/chunks.jsonl.

Usage: python scripts/build_chunks.py
Re-run after more PDFs are downloaded; it rebuilds the file from scratch.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import pymupdf

from niyam.ingest import chunk_circular

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "chunks.jsonl"
MIN_TEXT_CHARS = 200  # below this a PDF is probably a scan with no text layer


def main() -> int:
    circulars = [json.loads(line) for line in (RAW / "circulars.jsonl").read_text(encoding="utf-8").splitlines()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    chunks, scanned, missing = [], [], 0
    for meta in circulars:
        pdf = RAW / "pdf" / f"{meta['id']}.pdf"
        if not pdf.exists() or time.time() - pdf.stat().st_mtime < 5:  # skip files still being written
            missing += 1
            continue
        with pymupdf.open(pdf) as doc:
            text_chars = sum(len(page.get_text().strip()) for page in doc)
        if text_chars < MIN_TEXT_CHARS:
            scanned.append(meta["number"])  # still chunked: ingest.py OCRs pages without a text layer
        chunks.extend(chunk_circular(meta, pdf))

    with OUT.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    sizes = [len(c.text) for c in chunks]
    done = len(circulars) - missing
    print(f"circulars chunked: {done} | not downloaded yet: {missing} | OCR'd (no text layer): {len(scanned)}")
    if sizes:
        print(f"chunks: {len(chunks)} | chars per chunk: median {statistics.median(sizes):.0f}, max {max(sizes)}")
    for number in scanned:
        print(f"  OCR'd: {number}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
