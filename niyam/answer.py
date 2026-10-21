"""Grounded answers: turn retrieved passages into a prompt and stream a cited answer."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from typing import Protocol

from niyam.search import Hit

SYSTEM_PROMPT = """You answer questions about Indian GST using ONLY the numbered CBIC circular passages provided.

Rules:
- Cite every factual sentence with the number of the passage it comes from, in square brackets, e.g. [2].
- If the passages do not answer the question, say that you could not find it in the CBIC circulars and
  suggest confirming with a GST practitioner. Never fill gaps from outside knowledge.
- Answer in the language of the question (Hindi, English or Hinglish). Keep section numbers, rule numbers
  and form names exactly as written.
- Lead with the direct answer, then the conditions or exceptions that matter. Be brief.
- If a passage says a circular was amended, withdrawn or superseded, say so."""

CITATION = re.compile(r"\[(\d+)\]")


def page_label(hit: Hit) -> str:
    c = hit.chunk
    return f"page {c.page_start}" if c.page_start == c.page_end else f"pages {c.page_start}-{c.page_end}"


def build_prompt(question: str, hits: list[Hit]) -> str:
    sources = "\n\n".join(
        f"[{n}] Circular {h.chunk.number} dated {h.chunk.date}, {page_label(h)}\n"
        f"Subject: {h.chunk.subject}\n{h.chunk.text}"
        for n, h in enumerate(hits, 1)
    )
    return f"Passages:\n\n{sources}\n\nQuestion: {question}"


def cited_sources(answer: str, n_sources: int) -> list[int]:
    """Passage numbers the answer cites, in order, ignoring numbers that don't exist."""
    seen: list[int] = []
    for match in CITATION.finditer(answer):
        n = int(match.group(1))
        if 1 <= n <= n_sources and n not in seen:
            seen.append(n)
    return seen


class Generator(Protocol):
    def stream(self, system: str, prompt: str) -> Iterator[str]: ...


class GeminiGenerator:
    """Streams from Gemini. The key comes from GEMINI_API_KEY (put it in .env, never in code)."""

    def __init__(self, model: str | None = None):
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set; add it to .env (see .env.example)")
        self.client = genai.Client(api_key=api_key)
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        from google.genai import types

        config = types.GenerateContentConfig(system_instruction=system, temperature=0.1)
        for part in self.client.models.generate_content_stream(model=self.model, contents=prompt, config=config):
            if part.text:
                yield part.text
