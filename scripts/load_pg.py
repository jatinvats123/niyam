"""Load data/processed/chunks.jsonl and their e5 embeddings into Postgres.

Usage (with `docker compose up -d` running): python scripts/load_pg.py
"""

from __future__ import annotations

import time

from niyam.search import DenseIndex, load_chunks
from niyam.store import connect, load


def main() -> int:
    chunks = load_chunks()
    start = time.perf_counter()
    vectors = DenseIndex(chunks, "e5").vectors  # cached on disk after the first run
    with connect() as conn:
        load(conn, chunks, vectors)
        count = conn.execute("SELECT count(*), count(DISTINCT number) FROM chunks").fetchone()
    print(f"loaded {count[0]} chunks from {count[1]} circulars in {time.perf_counter() - start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
