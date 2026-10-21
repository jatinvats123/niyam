"""Turn circular PDFs into passages that can be searched and cited."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf

PAGE_HEADER = re.compile(r"^Page\s+\d+\s+of\s+\d+$", re.IGNORECASE)
# Numbered paragraphs: "2.", "3.1", "4.2." followed by text. Sub-points such as "(i)" stay with their parent.
PARA_START = re.compile(r"^\d{1,2}(?:\.\d{1,2})*\.?\s+\S")
SUBJECT = re.compile(r"^Sub(?:ject)?\s*[:.\-–]", re.IGNORECASE)
SENTENCE_END = re.compile(r"(?<=[.;:])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<circular id>-<n>"
    circular_id: int
    number: str  # "255/01/2026-GST"
    date: str  # "2026-06-25"
    subject: str
    page_start: int
    page_end: int
    text: str  # the passage itself; this is what a citation shows

    def embedding_text(self) -> str:
        # A short header restores the context a passage loses when it is cut out of its circular.
        return f"GST Circular {self.number} ({self.date}): {self.subject}\n\n{self.text}"


def clean(text: str) -> str:
    # NFKC folds ligatures ("ﬁ" -> "fi") and full-width forms; then collapse whitespace.
    return " ".join(unicodedata.normalize("NFKC", text).split())


OCR_CACHE = Path(__file__).resolve().parents[1] / "data" / "processed" / "ocr"
MIN_PAGE_CHARS = 50  # a page with less text than this is a scan without a text layer
_ocr_engine = None


def _ocr_page(page: pymupdf.Page, cache_file: Path) -> str:
    """OCR one scanned page with RapidOCR, caching the result (about 12 s per page on CPU)."""
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr import RapidOCR  # imported lazily: only 3 of 272 circulars need it

        _ocr_engine = RapidOCR()
    pix = page.get_pixmap(dpi=200)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    text = "\n".join(_ocr_engine(image).txts or [])
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(text, encoding="utf-8")
    return text


def _lines(pdf_path: Path) -> Iterator[tuple[int, str]]:
    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, 1):
            text = page.get_text()
            if len(text.strip()) < MIN_PAGE_CHARS:
                text = _ocr_page(page, OCR_CACHE / f"{pdf_path.stem}-p{page_no}.txt")
            for raw in text.splitlines():
                line = clean(raw)
                if line and not PAGE_HEADER.match(line):
                    yield page_no, line


def paragraphs(pdf_path: Path) -> list[tuple[int, str]]:
    """(page, paragraph) pairs in reading order, starting at the Subject line.

    The letterhead and addressee block before "Subject:" carry no answerable content. If a
    circular has no Subject line, everything is kept.
    """
    lines = list(_lines(pdf_path))
    start = next((i for i, (_, line) in enumerate(lines) if SUBJECT.match(line)), 0)
    paras: list[tuple[int, str]] = []
    current: list[str] = []
    current_page = lines[start][0] if lines else 1
    for page, line in lines[start:]:
        if current and PARA_START.match(line):
            paras.append((current_page, " ".join(current)))
            current = []
        if not current:
            current_page = page
        current.append(line)
    if current:
        paras.append((current_page, " ".join(current)))
    return paras


def _split_long(text: str, max_chars: int) -> list[str]:
    """Split an oversized paragraph at sentence boundaries into pieces of at most ~max_chars."""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    buf = ""
    sentences: list[str] = []
    for sentence in SENTENCE_END.split(text):
        if len(sentence) <= max_chars:
            sentences.append(sentence)
            continue
        # Table rows often run on without a full stop; fall back to word windows.
        words, window = sentence.split(), ""
        for word in words:
            if window and len(window) + 1 + len(word) > max_chars:
                sentences.append(window)
                window = word
            else:
                window = f"{window} {word}".strip()
        if window:
            sentences.append(window)
    for sentence in sentences:
        if buf and len(buf) + 1 + len(sentence) > max_chars:
            pieces.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        pieces.append(buf)
    return pieces


def chunk_circular(meta: dict, pdf_path: Path, max_chars: int = 1500) -> list[Chunk]:
    """Pack consecutive paragraphs into passages of up to ~max_chars, keeping page ranges."""
    chunks: list[Chunk] = []
    buf: list[str] = []
    pages: list[int] = []

    def flush() -> None:
        if buf:
            chunks.append(
                Chunk(
                    chunk_id=f"{meta['id']}-{len(chunks)}",
                    circular_id=meta["id"],
                    number=meta["number"],
                    date=meta["date"],
                    subject=meta["subject"],
                    page_start=min(pages),
                    page_end=max(pages),
                    text=" ".join(buf),
                )
            )
            buf.clear()
            pages.clear()

    for page, para in paragraphs(pdf_path):
        for piece in _split_long(para, max_chars):
            if buf and sum(len(b) + 1 for b in buf) + len(piece) > max_chars:
                flush()
            buf.append(piece)
            pages.append(page)
    flush()
    return chunks
