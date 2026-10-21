# Design decisions

Each entry: what was decided, why, and what was rejected. Numbers come from runs in this repo.

## Data

**Source: the CBIC Tax Information Portal API.** The old CBIC GST circulars page now returns 404. The new
portal is an Angular app whose JSON API lists all 272 GST circulars (2017–2026) and returns each PDF as
base64 (`niyam/cbic.py`). Rejected: scraping rendered HTML, which is slower and breaks on layout changes.

**TLS: verify through the OS trust store (`truststore`), never `verify=False`.** The portal serves an
incomplete certificate chain, so Python's bundled CA list rejects it while browsers (which fetch the
missing intermediate) accept it.

**Polite, resumable download.** One request at a time with a pause; already-downloaded PDFs are skipped,
so a failed or interrupted run is simply re-run. It is a public government service.

**English PDFs only, for now.** 77 of 272 circulars also have a Hindi PDF, but the download endpoint's
language code for Hindi is not known yet. Hindi questions are handled by multilingual retrieval over the
English text instead.

**Unicode NFKC normalisation everywhere.** CBIC's own titles and PDFs contain typographic ligatures
("Clariﬁcation" with a single "ﬁ" character), which silently break keyword search. NFKC folds them.

**Scanned circulars are flagged, not guessed at.** A PDF with under 200 characters of text has no text
layer and needs OCR (3 of the first 133).

## Chunking

**Split at numbered paragraphs, pack to ~1,500 characters.** Circulars are written as numbered
paragraphs ("2.", "3.1") with sub-points ("(i)", "(ii)"), so a paragraph is the natural unit of meaning.
Sub-points stay with their parent. Oversized paragraphs split at sentence ends, and run-on table rows at
word windows. Each passage keeps its page range for citations. Rejected: fixed-size windows, which cut
clarifications in half.

**Drop the letterhead and addressee block.** Everything before the "Subject:" line ("Government of
India… Madam/Sir") has nothing answerable in it and would match every query equally.

**Contextual header on every passage.** Before indexing, each passage is prefixed with
`GST Circular <number> (<date>): <subject>`, so a passage cut out of its circular still says what it is
about. The passage alone is what a citation shows.

## Evaluation

**A hand-labelled question set before any tuning.** `data/eval/questions.jsonl` started with 35 questions written
the way a small-business owner asks them (28 English, 7 Hindi or Hinglish), each labelled with the
circular(s) that answer it. Questions avoid copying the circular's subject line, so keyword overlap
doesn't make the test easy. A question counts as answered at k when a relevant circular is among the
first k distinct circulars retrieved.

**50 questions: 45 answerable, 5 out of scope.** The out-of-scope questions (income tax, company
registration, customs duty, PF withdrawal in Hindi) have no relevant circular; they test that the system
declines instead of inventing an answer, so they are scored by the answer eval, not by retrieval recall.

**Baseline first: BM25 keyword search** on all 272 circulars: R@1 0.76, R@5 0.89, MRR 0.82, but
Hindi/Hinglish R@5 only 0.50, because Devanagari words never appear in English circulars.

**Two gates in CI, sized to what they cost.** Every push runs the unit tests and a BM25 retrieval gate on
the committed passages (seconds, no models). A weekly and on-demand workflow downloads the models (cached),
gates BM25, e5 (production) and hybrid retrieval, then runs the answer eval with an LLM faithfulness judge. A gate fails
if recall@5 or MRR drops more than 0.03 below `data/eval/baseline.json`. If the `GEMINI_API_KEY` secret is
missing, the answer eval is skipped with a visible warning rather than passing silently.

**Answer metrics.** For answerable questions: the share whose answer cites at least one passage from a
relevant circular, and citation precision (cited passages that come from relevant circulars). For
out-of-scope questions: the share that decline with no citations. With `--judge`, a second model call
counts how many of the answer's claims its cited passages actually support (faithfulness).

**Answer eval: built, not run yet.** `scripts/eval_answers.py --judge` needs `GEMINI_API_KEY` in `.env`,
and no key was available on 22 Oct 2026, so there are no answer-quality or faithfulness numbers yet and
none are claimed. The same applies to measuring what the LangGraph rewrite and self-check add: run
`--flow plain` and `--flow graph` with `--judge` and compare.

**Labels are unambiguous.** CBIC reuses some labels: several corrigenda are just "Corrigendum", and two
different circulars are both "1/1/2017". Repeated labels get their date appended, so every citation
points at exactly one document.

## Retrieval set-up

All numbers: 45 answerable questions (37 English, 8 Hindi or Hinglish), all 272 circulars, 1,697 passages,
one laptop CPU, measured with `scripts/eval_retrieval.py` on 22 Oct 2026. Latency is per query and includes
embedding the query.

| Set-up | R@1 | R@3 | R@5 | MRR | Hindi R@5 | ms/query |
|---|---|---|---|---|---|---|
| BM25 (keyword) | 0.76 | 0.89 | 0.89 | 0.82 | 0.50 | 2 |
| potion-multilingual (static embeddings) | 0.71 | 0.89 | 0.89 | 0.80 | 0.75 | 7 |
| **multilingual-e5-large (dense)** | **0.96** | **1.00** | **1.00** | **0.98** | **1.00** | **~100** |
| hybrid: BM25 + e5, RRF | 0.93 | 0.98 | 0.98 | 0.96 | 1.00 | ~95 |
| e5 + MiniLM-L-12 reranker, top 8 | 0.91 | 1.00 | 1.00 | 0.96 | 1.00 | 1,589 |
| hybrid + MiniLM-L-12 reranker, top 8 | 0.89 | 0.98 | 0.98 | 0.93 | 1.00 | 1,715 |
| e5 + bge-reranker-v2-m3 (int8), top 8 | 0.96 | 1.00 | 1.00 | 0.98 | 1.00 | 5,543 |
| hybrid + bge-reranker-v2-m3 (int8), top 8 | 0.93 | 0.98 | 0.98 | 0.95 | 1.00 | 5,432 |
| Postgres: pgvector (HNSW) only | 0.96 | 1.00 | 1.00 | 0.98 | 1.00 | 113 |
| Postgres: full-text + pgvector, RRF | 0.89 | 1.00 | 1.00 | 0.94 | 1.00 | 146 |

Earlier run on the first 35 questions: hybrid + bge-reranker-v2-m3 over the top 30 candidates scored R@1
0.91, R@5 1.00, MRR 0.95 at about 35 seconds per query, so it was not run again on the full set.

**Production retriever: multilingual-e5-large alone.** It is the best set-up on every metric, including
Hindi and Hinglish (R@5 1.00 against BM25's 0.50), at about 0.1 s per query on a CPU.

**Hybrid search was dropped from the default path.** Adding BM25 through RRF cost 0.03 R@1 and 0.02 MRR:
on this question set keyword matches mostly promote passages that share common GST words with the
question, while e5 already ranks the right passage first. Hybrid stays in the eval and in CI so a future
question set with exact identifiers (circular or notification numbers) would show if it starts to help.

**No reranker.** Neither cheap option met the bar (better R@1 or MRR at under ~1.5 s per query). The
English MiniLM cross-encoder was both slower than the bar and worse (R@1 0.91 vs 0.96); bge-reranker-v2-m3
on only 8 candidates tied e5 exactly at about 54x its latency. A cross-encoder can only help when the right
passage is retrieved but ranked low, and e5 already has R@3 1.00, so there is almost nothing for it to fix.
Rejected: keeping the reranker "for quality" without evidence; it would make every answer 5-35 s slower.

**Postgres reproduces the in-memory results.** After `scripts/load_pg.py` (1,697 passages in 9 s), pgvector
with an HNSW index matches the in-memory e5 search on every metric and adds about 17 ms per query (113 vs
96 ms in the same run). The Postgres hybrid differs slightly from the in-memory hybrid (R@1 0.89 vs 0.93,
R@5 1.00 vs 0.98) because Postgres ranks keywords with `ts_rank_cd`, not BM25. The in-memory index is the
default because 1,697 vectors fit in 7 MB; Postgres is the path for a corpus that doesn't fit in memory or
for filtering by date and circular in SQL.

## Answering (API)

**Sources first, then the answer.** `POST /ask` streams Server-Sent Events: `sources` (the numbered
passages, each with circular number, date, pages and a link to the original PDF), then `token` events,
then `done` with the passage numbers the answer actually cited. The UI can show evidence before the
model has written a word.

**Grounding is enforced in two places.** The prompt allows only the numbered passages, requires a `[n]`
citation on every factual sentence, and requires saying "not found" rather than filling gaps. The server
then re-checks citations: numbers that don't match a provided passage are dropped. If retrieval returns
nothing, the model is never called.

**Answer in the question's language.** Hindi questions get Hindi answers, while section numbers and form
names stay as written, so they still match the circulars.

**The LLM sits behind a one-method interface (`stream`).** Gemini is the default (key from `.env`, never in code);
tests use a fake generator, so the API tests run in seconds with no key and no network. Without a key,
`/ask` answers 503 with the reason instead of a bare server error, and `/search` keeps working.

**`GET /search` needs no LLM.** Retrieval can be inspected and evaluated on its own.

## Answer flow (LangGraph)

**rewrite → retrieve → answer → self-check, as a LangGraph `StateGraph`** (`niyam/graph.py`). Each step is
a node over one typed state (question, search query, passages, answer, unsupported sentences, attempts),
and the two decisions are conditional edges: no passages ends the flow before the model is called, and an
answer the checker rejects goes back to `answer` once. Rejected: a hand-written chain of function calls,
which works for the straight path but hides the retry loop in `if` statements and is harder to extend
(a clarifying-question node, a per-node trace).

**Rewrite only questions that need it.** Hindi (any non-ASCII text) and Hinglish (a list of common
Hindi words in Latin script) are rewritten by the LLM into English search terms with the official
vocabulary ("input tax credit", "late fee"). English questions skip the call: e5 already retrieves them
with R@1 0.96, so a rewrite would add an LLM round trip for no measured gain.

**Search with both the original and the rewrite, fused with RRF.** The rewrite fixes vocabulary, but a
bad rewrite would lose a question that the multilingual model already handles (Hindi R@5 is 1.00 without
any rewrite). Fusing both rankings keeps whatever either one finds.

**Self-check every sentence against the passage it cites.** A second LLM call gets the passages and the
answer and returns JSON listing the factual sentences that their cited passage does not support (or that
cite nothing). If any are listed, the answer is regenerated once with those sentences named as feedback.
The regenerated answer is final and is not checked again: a third model call would add latency to every
bad answer and still could not loop forever. A checker reply that is not valid JSON is treated as "no
evidence against the answer" so a flaky checker never blocks an answer.

**Streaming survives the graph.** Nodes write `sources`, `token` and `retry` events to LangGraph's
custom stream, so `/ask` keeps its Server-Sent Events interface: the first token still arrives as soon
as the model produces it (checked with a slow fake generator: tokens arrive 0.5 s apart, not at the end).
When the self-check fails, a `retry` event tells the client to discard the draft before the second
answer streams. The API tests written before the graph existed pass unchanged.

## MCP server and UI

**MCP server over stdio** (`niyam/mcp_server.py`, official `mcp` SDK). It exposes `search_circulars(query,
k)` (no LLM key needed) and `ask(question)` (runs the same LangGraph flow). Outputs are typed
(`TypedDict`), so clients get structured results with circular number, date, pages and text for every
passage. The installed SDK is `mcp` 2.x, where `FastMCP` was renamed `MCPServer`; the code uses the new
name. Rejected: an HTTP transport, which a desktop client does not need and which would need auth.

**One static HTML page, no build step** (`niyam/static/index.html`, served at `/`). It reads the `/ask`
stream with `fetch` and a stream reader (EventSource only supports GET), renders `[n]` as links to the
source cards, highlights the passages the answer actually cited, and links each card to the original
PDF at the cited page. Dark mode follows the OS.

## Packaging

**Docker image with the model and embeddings baked in.** The build downloads the e5 model and loads the
passage embeddings (computing them if the cache is not in the build context), then loads them again with
`HF_HUB_OFFLINE=1` to prove the container needs no network to start. The circular PDFs are not in the
image; they are mounted read-only when PDF links are wanted. The package is installed editable inside
the image because the code finds `data/` next to the package directory.

**Two image-size traps, measured.** The first build was 12 GB: fastembed copies the 2.1 GB model out of
the Hugging Face blob store (two copies), and a `chown -R` after the download copied it into a third
layer. Creating the user and its directories before the download, and hard-linking the copy back to its
blob in the same layer, cut the model layer to 2.25 GB (Docker Desktop reports 4.78 GB for the image).

**Measured in the container (22 Oct 2026):** the first `/search` after start takes about 20 s (loading
e5), later ones 0.3 s including HTTP; memory is 2.6 GB after the model loads. Not deployed: that rules
out the free tiers of most hosts. Rejected for now: loading the model at start-up, which would only move
the 20 s into the health check.
