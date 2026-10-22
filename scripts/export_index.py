"""Export the passages and their e5 vectors to one file, so the API can serve search without Postgres and
without embedding all 1,697 passages at start-up (NIYAM_INDEX=file).

Usage: python scripts/export_index.py     # writes data/index/e5-passages.npz; re-run after build_chunks.py
"""

from __future__ import annotations

import time

import numpy as np

from niyam.search import INDEX_FILE, ROOT, DenseIndex, load_chunks, read_index, write_index


def main() -> int:
    start = time.perf_counter()
    index = DenseIndex(load_chunks(), "e5")  # embeds the passages, or loads them from the on-disk cache
    write_index(INDEX_FILE, index.chunks, index.vectors, "e5")
    key, chunks, vectors = read_index(INDEX_FILE)  # read it back the way the API will
    if chunks != index.chunks or not np.array_equal(vectors, index.vectors):
        print("export check FAILED: the file does not round-trip")
        return 1
    print(f"{INDEX_FILE.relative_to(ROOT).as_posix()}: {len(chunks)} passages, {vectors.shape[1]}-d {key} vectors "
          f"({vectors.dtype}), {INDEX_FILE.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
