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


def test_ask_without_llm_key_is_a_clear_503(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(api, "load_dotenv", lambda *_: None)  # don't pick up a developer's .env
    api.get_generator.cache_clear()
    api.app.dependency_overrides[api.get_retriever] = lambda: FakeRetriever(HITS)
    response = client.post("/ask", json={"question": "Can I claim ITC on demo cars?"})
    assert response.status_code == 503 and "GEMINI_API_KEY" in response.json()["detail"]
