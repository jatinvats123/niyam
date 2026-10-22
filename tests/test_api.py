import json

import pytest
from fastapi.testclient import TestClient

from niyam import api
from niyam.ingest import Chunk
from niyam.search import Hit


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, k=6):
        return self.hits[:k]


class FakeGenerator:
    def __init__(self, pieces):
        self.pieces = pieces
        self.prompts = []

    def stream(self, system, prompt):
        self.prompts.append(prompt)
        yield from self.pieces


HITS = [
    Hit(Chunk("7-0", 7, "231/25/2024-GST", "2024-09-11", "ITC on demo vehicles", 2, 3, "Demo vehicles ..."), 3.1),
    Hit(Chunk("9-1", 9, "140/10/2020 - GST", "2020-06-10", "Director's remuneration", 1, 1, "Directors ..."), -2.0),
]


def events(response):
    parsed = []
    for block in response.text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        parsed.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return parsed


@pytest.fixture
def client():
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def use(retriever, generator=None):
    api.app.dependency_overrides[api.get_retriever] = lambda: retriever
    api.app.dependency_overrides[api.get_generator] = lambda: generator or FakeGenerator([])


def test_search_returns_citable_sources(client):
    use(FakeRetriever(HITS))
    body = client.get("/search", params={"q": "ITC on demo cars", "k": 2}).json()
    assert body[0] == {"n": 1, "circular": "231/25/2024-GST", "date": "2024-09-11", "subject": "ITC on demo vehicles",
                       "pages": "pages 2-3", "text": "Demo vehicles ...", "score": 3.1, "pdf": "/circulars/7/pdf"}


def test_ask_streams_sources_then_tokens_then_citations(client):
    generator = FakeGenerator(["Yes, ITC is available ", "on demo vehicles [1]. ", "See also [7] and [1]."])
    use(FakeRetriever(HITS), generator)
    parsed = events(client.post("/ask", json={"question": "Can I claim ITC on demo cars?"}))
    names = [name for name, _ in parsed]
    assert names == ["sources", "token", "token", "token", "done"]
    assert parsed[-1][1] == {"cited": [1]}  # [7] doesn't exist, so it is dropped
    assert "[1] Circular 231/25/2024-GST dated 2024-09-11, pages 2-3" in generator.prompts[0]


def test_ask_without_matches_does_not_call_the_model(client):
    generator = FakeGenerator(["should not be used"])
    use(FakeRetriever([]), generator)
    parsed = events(client.post("/ask", json={"question": "What is the capital of France?"}))
    assert parsed[1][1]["text"].startswith("I couldn't find this")
    assert generator.prompts == []


def test_ask_validates_input(client):
    use(FakeRetriever(HITS))
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "k": 50}).status_code == 422


def test_pdf_404_for_unknown_circular(client):
    assert client.get("/circulars/999999999/pdf").status_code == 404


def test_web_ui_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "text/html" in response.headers["content-type"]
    assert 'fetch("/ask"' in response.text


def test_ask_without_llm_key_returns_passages_labelled_search_only(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(api, "load_dotenv", lambda *_: None)  # don't pick up a developer's .env
    api.get_generator.cache_clear()
    api.app.dependency_overrides[api.get_retriever] = lambda: FakeRetriever(HITS)
    try:
        parsed = events(client.post("/ask", json={"question": "Can I claim ITC on demo cars?"}))
        health = client.get("/health").json()
    finally:
        api.get_generator.cache_clear()
    assert [name for name, _ in parsed] == ["sources", "done"]  # no tokens: nothing is generated
    assert parsed[0][1][0] == {"n": 1, "circular": "231/25/2024-GST", "date": "2024-09-11",
                               "subject": "ITC on demo vehicles", "pages": "pages 2-3", "text": "Demo vehicles ...",
                               "score": 3.1, "pdf": "/circulars/7/pdf",
                               "cbic_url": "https://taxinformation.cbic.gov.in/view-pdf/7/ENG/Circulars"}
    assert parsed[1][1] == {"cited": [], "mode": "search_only"}
    assert health == {"status": "ok", "mode": "search_only"}


def test_with_a_key_health_says_answer_and_sources_carry_no_search_only_fields(client):
    use(FakeRetriever(HITS), FakeGenerator(["Yes [1]."]))
    assert client.get("/health").json() == {"status": "ok", "mode": "answer"}
    parsed = events(client.post("/ask", json={"question": "Can I claim ITC on demo cars?"}))
    assert "cbic_url" not in parsed[0][1][0] and parsed[-1] == ("done", {"cited": [1]})


def test_ask_over_the_rate_limit_is_a_429_with_retry_after(client):
    now = [1000.0]
    limiter = api.RateLimiter("2/minute", clock=lambda: now[0])
    use(FakeRetriever(HITS))
    api.app.dependency_overrides[api.get_generator] = lambda: None
    api.app.dependency_overrides[api.get_rate_limiter] = lambda: limiter

    def ask():
        return client.post("/ask", json={"question": "Can I claim ITC on demo cars?"})

    assert [ask().status_code, ask().status_code] == [200, 200]
    blocked = ask()
    assert blocked.status_code == 429 and blocked.headers["retry-after"] == "60"
    now[0] += 60
    assert ask().status_code == 200


def test_rate_limiter_keeps_each_window_and_client_apart():
    now = [0.0]
    limiter = api.RateLimiter("1/second, 3/day", clock=lambda: now[0])
    assert limiter.wait("1.2.3.4") == 0 and limiter.wait("5.6.7.8") == 0
    assert limiter.wait("1.2.3.4") == 1  # a second call within the same second
    for t in (1, 2):
        now[0] = t
        assert limiter.wait("1.2.3.4") == 0
    now[0] = 3
    assert limiter.wait("1.2.3.4") == 86_400 - 3  # fourth call today: wait until the first is a day old


def test_pdf_falls_back_to_the_official_copy_when_not_downloaded(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "PDF_DIR", tmp_path)  # no local PDFs, as in the Docker image and the Space
    response = client.get("/circulars/1003335/pdf", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "https://taxinformation.cbic.gov.in/view-pdf/1003335/ENG/Circulars"
