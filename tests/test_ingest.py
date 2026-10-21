from pathlib import Path

import pymupdf

from niyam.ingest import _split_long, chunk_circular, clean, paragraphs


def make_pdf(path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text, fontsize=10)
    doc.save(path)
    return path


CIRCULAR_PAGES = [
    "Page 1 of 2\nCircular No. 1/1/2026-GST\nGovernment of India\nMadam/Sir,\n"
    "Subject: Clarification on demo vehicles - reg.\n"
    "References have been received seeking clarification.\n"
    "2. The issue is whether credit is available:\n(i) on demo vehicles; and\n(ii) on spares.\n",
    "Page 2 of 2\n3. It is clarified that credit is available.\n",
]
META = {"id": 7, "number": "1/1/2026-GST", "date": "2026-01-01", "subject": "Demo vehicles"}


def test_clean_folds_ligatures_and_whitespace():
    assert clean("Clariﬁcation   of\tﬂow") == "Clarification of flow"


def test_paragraphs_start_at_subject_and_keep_sub_points(tmp_path):
    paras = paragraphs(make_pdf(tmp_path / "c.pdf", CIRCULAR_PAGES))
    texts = [text for _, text in paras]
    assert texts[0].startswith("Subject: Clarification on demo vehicles")
    assert not any("Government of India" in t or "Page 1 of 2" in t for t in texts)
    assert "(i) on demo vehicles; and (ii) on spares." in texts[1]  # sub-points stay with paragraph 2
    assert paras[-1] == (2, "3. It is clarified that credit is available.")


def test_chunks_carry_page_ranges_and_context_header(tmp_path):
    chunks = chunk_circular(META, make_pdf(tmp_path / "c.pdf", CIRCULAR_PAGES), max_chars=120)
    assert [c.chunk_id for c in chunks] == [f"7-{i}" for i in range(len(chunks))]
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 2
    assert all(len(c.text) <= 120 for c in chunks)
    assert chunks[0].embedding_text().startswith("GST Circular 1/1/2026-GST (2026-01-01): Demo vehicles")


def test_split_long_respects_limit_even_without_full_stops():
    sentences = "This is a sentence. " * 20
    table_row = "word " * 100  # a run-on table row with no full stop
    for text in (sentences, table_row):
        pieces = _split_long(text.strip(), 80)
        assert all(len(p) <= 80 for p in pieces)
        assert " ".join(pieces).split() == text.split()
