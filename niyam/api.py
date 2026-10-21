"""HTTP API: search passages, ask a question (streamed, with citations), fetch the source PDF.

Run: uvicorn niyam.api:app --port 8000
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from niyam.answer import Generator, GeminiGenerator, page_label
from niyam.graph import AnswerFlow
from niyam.search import Hit, Retriever

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "raw" / "pdf"
UI_PAGE = Path(__file__).parent / "static" / "index.html"
NOT_FOUND = ("I couldn't find this in the CBIC GST circulars I have. "
             "Please confirm with a GST practitioner.")


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()  # loads the passages and the e5 model and embeddings once


@lru_cache(maxsize=1)
def get_generator() -> Generator:
    try:
        return GeminiGenerator()
    except RuntimeError as exc:  # no API key: say so instead of a bare 500
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_dotenv(ROOT / ".env")
    yield


app = FastAPI(title="Niyam", description="GST answers from CBIC circulars, with citations", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=6, ge=1, le=10)


def source_json(n: int, hit: Hit) -> dict:
    c = hit.chunk
    return {"n": n, "circular": c.number, "date": c.date, "subject": c.subject, "pages": page_label(hit),
            "text": c.text, "score": round(hit.score, 4), "pdf": f"/circulars/{c.circular_id}/pdf"}


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(UI_PAGE, media_type="text/html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/search")
def search(q: str = Query(min_length=3, max_length=500), k: int = Query(default=6, ge=1, le=10),
           retriever: Retriever = Depends(get_retriever)) -> list[dict]:
    return [source_json(n, h) for n, h in enumerate(retriever.search(q, k), 1)]


def get_flow(retriever: Retriever = Depends(get_retriever), generator: Generator = Depends(get_generator)) -> AnswerFlow:
    return AnswerFlow(retriever, generator)


@app.post("/ask")
def ask(body: AskRequest, flow: AnswerFlow = Depends(get_flow)) -> StreamingResponse:
    """Streams `sources`, then `token`s, then `done` with the cited passage numbers. If the self-check
    finds unsupported sentences, a `retry` event says the answer so far is discarded and a new one follows."""

    def events() -> Iterator[str]:
        try:
            for event, payload in flow.stream(body.question, body.k):
                if event == "sources":
                    yield sse("sources", [source_json(n, h) for n, h in enumerate(payload, 1)])
                    if not payload:
                        yield sse("token", {"text": NOT_FOUND})
                elif event == "token":
                    yield sse("token", {"text": payload})
                elif event == "retry":
                    yield sse("retry", {"unsupported": payload})
                else:
                    yield sse("done", {"cited": payload})
        except Exception as exc:  # report upstream LLM failures to the client instead of a broken stream
            yield sse("error", {"message": f"answer generation failed: {exc}"})

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/circulars/{circular_id}/pdf")
def circular_pdf(circular_id: int) -> FileResponse:
    path = PDF_DIR / f"{circular_id}.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="circular not found")
    return FileResponse(path, media_type="application/pdf", filename=path.name)
