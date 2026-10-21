"""Retrieval over circular passages: BM25, dense vectors, hybrid fusion and reranking.

All set-ups stay here so the eval can compare them; `Retriever` is the one the API uses.
This in-memory version is the reference used for evaluation. The Postgres version
(full-text search + pgvector) should reproduce its rankings.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import bm25s
import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from niyam.ingest import Chunk

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.jsonl"
CACHE_DIR = ROOT / "data" / "processed"

# key -> (fastembed model name, query prefix, passage prefix). E5 models are trained with these prefixes.
EMBEDDERS = {
    "potion": ("minishlab/potion-multilingual-128M", "", ""),
    "e5": ("intfloat/multilingual-e5-large", "query: ", "passage: "),
}
RERANKER = "BAAI/bge-reranker-v2-m3-int8"
MINILM_RERANKER = "Xenova/ms-marco-MiniLM-L-12-v2"  # small and fast, but English-only
RRF_K = 60  # standard constant for reciprocal rank fusion


def load_chunks(path: Path = CHUNKS_FILE) -> list[Chunk]:
    return [Chunk(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]


@dataclass
class Hit:
    chunk: Chunk
    score: float


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.retriever = bm25s.BM25()
        self.retriever.index(bm25s.tokenize([c.embedding_text() for c in chunks], stopwords="en", show_progress=False),
                             show_progress=False)

    def search(self, query: str, k: int = 50) -> list[Hit]:
        tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        if not tokens.vocab:  # e.g. a query in Devanagari has no tokens in common with English text
            return []
        k = min(k, len(self.chunks))
        ids, scores = self.retriever.retrieve(tokens, k=k, show_progress=False)
        return [Hit(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if s > 0]


class DenseIndex:
    def __init__(self, chunks: list[Chunk], key: str = "e5"):
        model_name, self.query_prefix, passage_prefix = EMBEDDERS[key]
        self.chunks = chunks
        self.model = TextEmbedding(model_name)
        digest = hashlib.sha1("".join(c.chunk_id for c in chunks).encode()).hexdigest()[:10]
        cache = CACHE_DIR / f"emb-{key}-{digest}.npy"
        if cache.exists():
            self.vectors = np.load(cache)
        else:
            texts = [passage_prefix + c.embedding_text() for c in chunks]
            self.vectors = np.array(list(self.model.embed(texts, batch_size=16)), dtype=np.float32)
            np.save(cache, self.vectors)
        self.vectors /= np.linalg.norm(self.vectors, axis=1, keepdims=True)

    def embed_query(self, query: str) -> np.ndarray:
        q = np.array(next(iter(self.model.embed([self.query_prefix + query]))), dtype=np.float32)
        return q / np.linalg.norm(q)

    def search(self, query: str, k: int = 50) -> list[Hit]:
        sims = self.vectors @ self.embed_query(query)
        top = np.argsort(-sims)[:k]
        return [Hit(self.chunks[i], float(sims[i])) for i in top]


def reciprocal_rank_fusion(*rankings: list[Hit], k: int = 50) -> list[Hit]:
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            cid = hit.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            by_id[cid] = hit.chunk
    ordered = sorted(scores, key=scores.get, reverse=True)[:k]
    return [Hit(by_id[cid], scores[cid]) for cid in ordered]


class Reranker:
    def __init__(self, model_name: str = RERANKER):
        self.model = TextCrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit], top_n: int = 30) -> list[Hit]:
        candidates = hits[:top_n]
        scores = list(self.model.rerank(query, [h.chunk.embedding_text() for h in candidates]))
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [Hit(h.chunk, float(s)) for h, s in ranked] + hits[top_n:]


class Retriever:
    """The production pipeline: multilingual-e5-large dense search alone.

    On the 45 answerable eval questions it beats BM25 + e5 fusion (R@1 0.96 vs 0.93) and every reranker
    tried, which either tied it at 50x the latency or lowered R@1 (see DESIGN.md, "Retrieval set-up").
    """

    def __init__(self, chunks: list[Chunk] | None = None):
        self.dense = DenseIndex(chunks if chunks is not None else load_chunks(), "e5")

    def search(self, query: str, k: int = 6) -> list[Hit]:
        return self.dense.search(query, k)


def circular_ranking(hits: list[Hit]) -> list[str]:
    """Circular numbers in rank order, each listed once (an answer cites circulars, not chunks)."""
    seen: list[str] = []
    for hit in hits:
        if hit.chunk.number not in seen:
            seen.append(hit.chunk.number)
    return seen
