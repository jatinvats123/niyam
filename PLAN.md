# Niyam: plan and status

Hindi/English assistant that answers GST and compliance questions for small businesses from official CBIC
circulars, with citations. This is the "missing Python RAG project" from the resume audit
(https://claude.ai/artifact/1LUW1zMDt1knu1ndDfecHH, section "Missing project").

## Status (22 Oct 2026)

- Built: downloader (all 272 circulars), ingestion (1,697 passages, OCR fallback), BM25 / e5 / hybrid /
  reranker retrieval with a 50-question eval and CI gates, Postgres + pgvector store, FastAPI with SSE
  answers, LangGraph flow (rewrite → retrieve → answer → self-check), MCP server, web UI, Dockerfile.
- Production retrieval is e5 alone: R@1 0.96, R@5 1.00, MRR 0.98, Hindi R@5 1.00, ~0.1 s/query. Hybrid
  and rerankers were measured and rejected (DESIGN.md, "Retrieval set-up").
- Not done: the answer eval (`scripts/eval_answers.py --judge`) needs `GEMINI_API_KEY` in `.env`;
  deployment; the 90-second demo; pushing to GitHub.

## Build order (days 4–10 of the plan)

1. Downloader + ingestion: 200–500 circulars → PyMuPDF text → chunks with circular number, date and
   subject as metadata.
2. Postgres + pgvector via docker compose (`pgvector/pgvector` image); multilingual embeddings with
   fastembed; Postgres full-text index for hybrid search.
3. Retrieval: hybrid (full-text + vector) with a cross-encoder reranker. Hand-write 20 questions with
   their source circulars and measure recall@5 before and after reranking.
4. FastAPI service: streaming answers with citations (LLM key goes in `.env`, never in git).
5. Evals: 50-question golden set, faithfulness and context precision, gated in GitHub Actions.
6. LangGraph flow: rewrite query → retrieve → rerank → answer → self-check. Re-measure.
7. MCP server exposing `search_circulars` and `ask`; small web UI; deploy; 90-second demo; README in
   Pillion's format.

Keep a DESIGN.md of every decision and trade-off as you go: interviewers will ask about each one.
Only after step 7, add the project and Python / FastAPI / RAG / pgvector / MCP evidence to the AI resume.
