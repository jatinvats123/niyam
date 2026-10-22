---
title: Niyam
emoji: 📜
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Cited search over all 272 CBIC GST circulars
---

# Niyam (नियम)

Ask a GST question in English, Hindi or Hinglish and get the passages from the 272 official CBIC GST
circulars that answer it, each with its circular number, date, pages and a link to the official PDF.

This Space runs **search only (no AI answer)**: it has no LLM key, so it shows the best-matching passages
instead of a written answer. With a `GEMINI_API_KEY` secret it streams answers that cite those passages,
self-checked sentence by sentence. `/ask` is rate-limited per visitor (`NIYAM_ASK_RATE_LIMIT`, default
10 a minute and 100 a day).

Code, evals and design notes: https://github.com/jatinvats123/niyam. The `Dockerfile` here builds the app
from that repository.
