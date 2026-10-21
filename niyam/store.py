"""Postgres storage and hybrid search: full-text (tsvector) + pgvector, fused with reciprocal rank fusion."""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from niyam.ingest import Chunk
from niyam.search import RRF_K, Hit

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://niyam:niyam@localhost:5433/niyam")
DIM = 1024  # intfloat/multilingual-e5-large

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      text PRIMARY KEY,
    circular_id   bigint  NOT NULL,
    number        text    NOT NULL,
    circular_date date,
    subject       text    NOT NULL,
    page_start    int     NOT NULL,
    page_end      int     NOT NULL,
    body          text    NOT NULL,   -- the passage shown as a citation
    search_text   text    NOT NULL,   -- header + passage: what is indexed
    tsv           tsvector GENERATED ALWAYS AS (to_tsvector('english', search_text)) STORED,
    embedding     vector({DIM})
);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
"""

# plainto_tsquery joins terms with AND, which makes a full question match almost nothing.
# Turning '&' into '|' gives BM25-like OR semantics; ts_rank_cd still rewards passages matching more terms.
HYBRID_SQL = f"""
WITH q AS (
    SELECT replace(plainto_tsquery('english', %(query)s)::text, '&', '|')::tsquery AS tsq
), kw AS (
    SELECT c.chunk_id, row_number() OVER (ORDER BY ts_rank_cd(c.tsv, q.tsq) DESC) AS r
    FROM chunks c, q
    WHERE q.tsq::text <> '' AND c.tsv @@ q.tsq
    ORDER BY ts_rank_cd(c.tsv, q.tsq) DESC
    LIMIT %(k)s
), vec AS (
    SELECT chunk_id, row_number() OVER (ORDER BY embedding <=> %(qvec)s) AS r
    FROM chunks
    ORDER BY embedding <=> %(qvec)s
    LIMIT %(k)s
)
SELECT c.chunk_id, c.circular_id, c.number, c.circular_date::text, c.subject, c.page_start, c.page_end, c.body,
       COALESCE(1.0 / ({RRF_K} + kw.r), 0) + COALESCE(1.0 / ({RRF_K} + vec.r), 0) AS score
FROM chunks c
LEFT JOIN kw USING (chunk_id)
LEFT JOIN vec USING (chunk_id)
WHERE kw.chunk_id IS NOT NULL OR vec.chunk_id IS NOT NULL
ORDER BY score DESC
LIMIT %(k)s
"""

VECTOR_SQL = """
SELECT chunk_id, circular_id, number, circular_date::text, subject, page_start, page_end, body,
       1 - (embedding <=> %(qvec)s) AS score
FROM chunks
ORDER BY embedding <=> %(qvec)s
LIMIT %(k)s
"""


def connect(url: str = DATABASE_URL) -> psycopg.Connection:
    conn = psycopg.connect(url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def load(conn: psycopg.Connection, chunks: list[Chunk], vectors: np.ndarray) -> None:
    """Replace the table contents with these chunks and their (normalised) embeddings."""
    conn.execute(SCHEMA)
    with conn.transaction():
        conn.execute("TRUNCATE chunks")
        with conn.cursor().copy(
            "COPY chunks (chunk_id, circular_id, number, circular_date, subject, page_start, page_end, body,"
            " search_text, embedding) FROM STDIN WITH (FORMAT BINARY)"
        ) as copy:
            copy.set_types(["text", "int8", "text", "date", "text", "int4", "int4", "text", "text", "vector"])
            for chunk, vector in zip(chunks, vectors, strict=True):
                day = date.fromisoformat(chunk.date) if chunk.date else None
                copy.write_row((chunk.chunk_id, chunk.circular_id, chunk.number, day, chunk.subject,
                                chunk.page_start, chunk.page_end, chunk.text, chunk.embedding_text(), vector))
    conn.execute("ANALYZE chunks")


def hybrid_search(conn: psycopg.Connection, query: str, query_vector: np.ndarray, k: int = 50) -> list[Hit]:
    return _hits(conn.execute(HYBRID_SQL, {"query": query, "qvec": query_vector, "k": k}).fetchall())


def vector_search(conn: psycopg.Connection, query_vector: np.ndarray, k: int = 50) -> list[Hit]:
    """pgvector only (HNSW index): the Postgres counterpart of the in-memory e5 search."""
    return _hits(conn.execute(VECTOR_SQL, {"qvec": query_vector, "k": k}).fetchall())


def _hits(rows: list[tuple]) -> list[Hit]:
    return [
        Hit(Chunk(chunk_id=r[0], circular_id=r[1], number=r[2], date=r[3] or "", subject=r[4],
                  page_start=r[5], page_end=r[6], text=r[7]), float(r[8]))
        for r in rows
    ]
