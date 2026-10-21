"""Download the GST circular list and every English PDF from the CBIC portal.

Resumable: PDFs already on disk are skipped, so the script can be stopped and re-run.
One request at a time with a pause, because the portal is a public government service.

Usage:
    python scripts/download_circulars.py              # everything, newest first
    python scripts/download_circulars.py --limit 20   # just the 20 newest missing PDFs
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from niyam.cbic import Circular, download_pdf, list_circulars, make_client

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PDF_DIR = RAW / "pdf"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="download at most N missing PDFs (newest first)")
    parser.add_argument("--pause", type=float, default=1.0, help="seconds to wait between requests")
    args = parser.parse_args()

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str]] = []
    with make_client() as client:
        rows = list_circulars(client)
        (RAW / "circulars.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        circulars = sorted((Circular.from_api(r) for r in rows), key=lambda c: c.date, reverse=True)
        # CBIC reuses some labels (two different circulars are both "1/1/2017"); add the date so citations are unambiguous.
        counts = Counter(c.number for c in circulars)
        circulars = [replace(c, number=f"{c.number} dated {c.date}") if counts[c.number] > 1 else c for c in circulars]
        (RAW / "circulars.jsonl").write_text(
            "".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in circulars), encoding="utf-8"
        )

        todo = [c for c in circulars if not (PDF_DIR / f"{c.id}.pdf").exists()][: args.limit]
        print(f"{len(circulars)} circulars listed; {len(todo)} PDFs to download", flush=True)
        for n, c in enumerate(todo, 1):
            for attempt in range(3):
                try:
                    pdf = download_pdf(client, c.id)
                except Exception as exc:  # network hiccups on a slow public server
                    if attempt == 2:
                        failures.append((c.number, str(exc)))
                        print(f"[{n}/{len(todo)}] FAILED {c.number}: {exc}", flush=True)
                    else:
                        time.sleep(5 * (attempt + 1))
                    continue
                (PDF_DIR / f"{c.id}.pdf").write_bytes(pdf)
                print(f"[{n}/{len(todo)}] {c.number} ({c.date}) {len(pdf) // 1024} KB", flush=True)
                break
            time.sleep(args.pause)

    print(f"done: {len(todo) - len(failures)} downloaded, {len(failures)} failed", flush=True)
    for number, error in failures:
        print(f"  failed {number}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
