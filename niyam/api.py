"""HTTP API: search passages, ask a question (streamed, with citations), fetch the source PDF.

Run: uvicorn niyam.api:app --port 8000
Without GEMINI_API_KEY, /ask is search-only: it streams the best passages and no AI answer.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from niyam.answer import Generator, GeminiGenerator, page_label
from niyam.cbic import view_url
from niyam.graph import AnswerFlow
from niyam.search import Hit, Retriever

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "raw" / "pdf"
CIRCULARS_FILE = ROOT / "data" / "raw" / "circulars.jsonl"
UI_PAGE = Path(__file__).parent / "static" / "index.html"
NOT_FOUND = ("I couldn't find this in the CBIC GST circulars I have. "
             "Please confirm with a GST practitioner.")
SEARCH_ONLY = "search_only"  # the /ask mode without an LLM key: passages, no AI answer
PERIODS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()  # loads the passages and the e5 model and embeddings once


@lru_cache(maxsize=1)
def get_generator() -> Generator | None:
    """Gemini, or None when GEMINI_API_KEY is not set: /ask then returns passages instead of an error."""
    return GeminiGenerator() if os.environ.get("GEMINI_API_KEY") else None


class RateLimiter:
    """Sliding-window limits per client address, kept in memory (the API runs as one process).

    `spec` lists the limits, e.g. "10/minute,100/day".
    """

    def __init__(self, spec: str, clock: Callable[[], float] = time.monotonic):
        self.limits = []
        for rule in filter(None, (part.strip() for part in spec.split(","))):
            count, _, period = rule.partition("/")
            self.limits.append((int(count), PERIODS[period.strip().removesuffix("s")]))
        self.horizon = max((period for _, period in self.limits), default=0)
        self.clock = clock
        self.calls: dict[str, deque[float]] = {}
        self.lock = threading.Lock()

    def wait(self, client: str) -> float:
        """0 if `client` may call now (the call is counted), else the seconds until it may."""
        now = self.clock()
        with self.lock:
            calls = self.calls.setdefault(client, deque())
            while calls and calls[0] <= now - self.horizon:
                calls.popleft()
            for count, period in self.limits:
                recent = [t for t in calls if t > now - period]
                if len(recent) >= count:
                    return recent[-count] + period - now
            calls.append(now)
            if len(self.calls) > 10_000:  # forget clients with no calls left in any window
                self.calls = {c: q for c, q in self.calls.items() if q and q[-1] > now - self.horizon}
            return 0.0


@lru_cache(maxsize=1)
def get_rate_limiter() -> RateLimiter | None:
    """Per-client limit on /ask from NIYAM_ASK_RATE_LIMIT (e.g. "10/minute,100/day"); unset means none."""
    spec = os.environ.get("NIYAM_ASK_RATE_LIMIT", "").strip()
    return RateLimiter(spec) if spec else None


def limit_ask(request: Request, limiter: RateLimiter | None = Depends(get_rate_limiter)) -> None:
    if limiter is None:
        return
    # Behind a proxy (a Hugging Face Space), uvicorn --proxy-headers puts the X-Forwarded-For client here.
    wait = limiter.wait(request.client.host if request.client else "unknown")
    if wait:
        seconds = math.ceil(wait)
        raise HTTPException(status_code=429, headers={"Retry-After": str(seconds)},
                            detail=f"Too many questions from your address; try again in {seconds} s.")


@lru_cache(maxsize=1)
def circular_ids() -> frozenset[int]:
    """Portal ids of the indexed circulars (their list is committed; the PDFs are not)."""
    if not CIRCULARS_FILE.exists():
        return frozenset()
    lines = CIRCULARS_FILE.read_text(encoding="utf-8").splitlines()
    return frozenset(json.loads(line)["id"] for line in lines if line.strip())


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_dotenv(ROOT / ".env")
    get_rate_limiter()  # a malformed NIYAM_ASK_RATE_LIMIT fails at start-up, not on the first question
    if os.environ.get("NIYAM_PRELOAD") == "1":  # load e5 and the passages before serving, not on the first query
        get_retriever().search("GST", 1)  # one query too: the model's first inference is the slow one
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
def health(generator: Generator | None = Depends(get_generator)) -> dict:
    return {"status": "ok", "mode": "answer" if generator else SEARCH_ONLY}


@app.get("/search")
def search(q: str = Query(min_length=3, max_length=500), k: int = Query(default=6, ge=1, le=10),
           retriever: Retriever = Depends(get_retriever)) -> list[dict]:
    return [source_json(n, h) for n, h in enumerate(retriever.search(q, k), 1)]


def search_only(retriever: Retriever, body: AskRequest) -> Iterator[str]:
    try:
        hits = retriever.search(body.question, body.k)
    except Exception as exc:
        yield sse("error", {"message": f"search failed: {exc}"})
        return
    yield sse("sources", [source_json(n, h) | {"cbic_url": view_url(h.chunk.circular_id)}
                          for n, h in enumerate(hits, 1)])
    yield sse("done", {"cited": [], "mode": SEARCH_ONLY})


@app.post("/ask", dependencies=[Depends(limit_ask)])
def ask(body: AskRequest, retriever: Retriever = Depends(get_retriever),
        generator: Generator | None = Depends(get_generator)) -> StreamingResponse:
    """Streams `sources`, then `token`s, then `done` with the cited passage numbers. If the self-check
    finds unsupported sentences, a `retry` event says the answer so far is discarded and a new one follows.

    Without GEMINI_API_KEY it is search-only: `sources` (each with `cbic_url`, the official PDF), then
    `done` with `{"cited": [], "mode": "search_only"}`, and no tokens."""
    if generator is None:
        return StreamingResponse(search_only(retriever, body), media_type="text/event-stream")
    flow = AnswerFlow(retriever, generator)

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
def circular_pdf(circular_id: int) -> Response:
    """The PDF from data/raw/pdf/ if it was downloaded, otherwise a redirect to the official copy."""
    path = PDF_DIR / f"{circular_id}.pdf"
    if path.exists():
        return FileResponse(path, media_type="application/pdf", filename=path.name)
    if circular_id in circular_ids():
        return RedirectResponse(view_url(circular_id), status_code=307)
    raise HTTPException(status_code=404, detail="circular not found")
