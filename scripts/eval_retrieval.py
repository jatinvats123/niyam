"""Compare retrieval set-ups on the hand-labelled question set, and gate CI on regressions.

A question counts as answered at k when one of its relevant circulars is among the first k
distinct circulars retrieved. Reports recall@1/3/5, MRR, Hindi/Hinglish recall@5 and latency.
Questions with no relevant circular (out-of-scope) are left to the answer eval.

Usage:
    python scripts/eval_retrieval.py                          # all set-ups
    python scripts/eval_retrieval.py --configs bm25 --gate    # fail if worse than data/eval/baseline.json
    python scripts/eval_retrieval.py --save-baseline          # record current numbers as the baseline
    python scripts/eval_retrieval.py --configs e5,pg-e5,file  # in memory vs Postgres vs the exported index file
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from niyam.search import (INDEX_FILE, MINILM_RERANKER, RERANKER, BM25Index, DenseIndex, Reranker, circular_ranking,
                          load_chunks, reciprocal_rank_fusion)

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data" / "eval" / "questions.jsonl"
BASELINE = ROOT / "data" / "eval" / "baseline.json"
TOLERANCE = 0.03  # allowed drop in recall@5 or MRR before the gate fails


def first_relevant_rank(ranking: list[str], relevant: set[str]) -> int | None:
    return next((i + 1 for i, number in enumerate(ranking) if number in relevant), None)


def recall(k: int, ranks: list[int | None]) -> float:
    return sum(1 for r in ranks if r and r <= k) / max(len(ranks), 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="bm25,potion,e5,hybrid,e5+rerank8")
    parser.add_argument("--gate", action="store_true", help="exit 1 if any config regresses against the baseline")
    parser.add_argument("--save-baseline", action="store_true", help="write these numbers to data/eval/baseline.json")
    args = parser.parse_args()
    configs = args.configs.split(",")

    chunks = load_chunks()
    corpus = {c.number for c in chunks}
    questions = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    answerable = [q for q in questions if set(q["relevant"]) & corpus]
    print(f"{len(chunks)} chunks from {len(corpus)} circulars | {len(answerable)} answerable questions "
          f"({len(questions) - len(answerable)} out-of-scope left to the answer eval)")

    bm25 = BM25Index(chunks)
    dense = {key: DenseIndex(chunks, key) for key in ("potion", "e5")
             if any(key in c for c in configs) or (key == "e5" and any("hybrid" in c or "pg-" in c for c in configs))}
    if any(c.split("+", 1)[0] == "file" for c in configs):  # the exported index (NIYAM_INDEX=file)
        dense["file"] = DenseIndex.from_file(INDEX_FILE, model=dense["e5"].model if "e5" in dense else None)
    # "+rerank" reranks the top 30 with bge-reranker-v2-m3; "+rerank8" only the top 8;
    # "+minilm8" the top 8 with the small English MiniLM cross-encoder.
    rerankers = {name: Reranker(model) for name, model in (("rerank", RERANKER), ("minilm", MINILM_RERANKER))
                 if any(f"+{name}" in c for c in configs)}
    conn = None
    if any(c.startswith("pg-") for c in configs):
        from niyam.store import connect, hybrid_search, vector_search

        conn = connect()

    def run(config: str, query: str):
        if config == "bm25":
            return bm25.search(query)
        base = config.split("+", 1)[0]
        if base in dense:
            fused = dense[base].search(query)
        elif base == "pg-e5":
            fused = vector_search(conn, dense["e5"].embed_query(query))
        elif base == "pg-hybrid":
            fused = hybrid_search(conn, query, dense["e5"].embed_query(query))
        else:
            fused = reciprocal_rank_fusion(bm25.search(query), dense["e5"].search(query))
        if "+" not in config:
            return fused
        name, top_n = re.fullmatch(r"([a-z]+)(\d*)", config.split("+", 1)[1]).groups()
        return rerankers[name].rerank(query, fused, top_n=int(top_n or 30))

    print(f"\n{'config':18}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'MRR':>7}{'Hi R@5':>8}{'ms/q':>8}")
    results: dict[str, dict] = {}
    misses: dict[str, list[str]] = {}
    for config in configs:
        ranks, hindi, elapsed = [], [], 0.0
        for q in answerable:
            start = time.perf_counter()
            ranking = circular_ranking(run(config, q["q"]))
            elapsed += time.perf_counter() - start
            rank = first_relevant_rank(ranking, set(q["relevant"]))
            ranks.append(rank)
            if q["lang"] != "en":
                hindi.append(rank)
            if rank is None or rank > 5:
                misses.setdefault(config, []).append(q["id"])
        mrr = sum(1 / r for r in ranks if r) / len(ranks)
        results[config] = {"r1": recall(1, ranks), "r3": recall(3, ranks), "r5": recall(5, ranks), "mrr": mrr,
                           "hi_r5": recall(5, hindi), "ms_per_query": 1000 * elapsed / len(ranks)}
        r = results[config]
        print(f"{config:18}{r['r1']:7.2f}{r['r3']:7.2f}{r['r5']:7.2f}{r['mrr']:7.2f}{r['hi_r5']:8.2f}"
              f"{r['ms_per_query']:8.0f}")
    for config, ids in misses.items():
        print(f"  missed@5 {config}: {', '.join(ids)}")

    if args.save_baseline:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
        baseline.update({c: {k: round(v, 4) for k, v in r.items() if k in ("r5", "mrr")} for c, r in results.items()})
        BASELINE.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        print(f"baseline saved for: {', '.join(results)}")

    if args.gate:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        failures = [f"{c} {metric} {results[c][metric]:.2f} < baseline {base[metric]:.2f} - {TOLERANCE}"
                    for c, base in baseline.items() if c in results
                    for metric in ("r5", "mrr") if results[c][metric] < base[metric] - TOLERANCE]
        for failure in failures:
            print(f"GATE FAILED: {failure}")
        print("gate: " + ("FAIL" if failures else "pass"))
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
