import json

from fastapi.testclient import TestClient

from niyam import api
from niyam.graph import CHECK_PROMPT, REWRITE_PROMPT, AnswerFlow, needs_rewrite, parse_unsupported
from niyam.ingest import Chunk
from niyam.search import Hit


def hit(chunk_id: str, number: str) -> Hit:
    return Hit(Chunk(chunk_id, 1, number, "2024-09-11", "ITC on demo vehicles", 2, 2, "Demo vehicles ..."), 1.0)


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def search(self, query, k=6):
        self.queries.append(query)
        return self.hits[:k]


class ScriptedGenerator:
    """Replies by role: rewrites, checker verdicts and answers each come from their own queue."""

    def __init__(self, answers, verdicts=(), rewrites=()):
        self.queues = {"answer": list(answers), CHECK_PROMPT: list(verdicts), REWRITE_PROMPT: list(rewrites)}
        self.calls = []

    def stream(self, system, prompt):
        role = system if system in (CHECK_PROMPT, REWRITE_PROMPT) else "answer"
        self.calls.append((role, prompt))
        yield from self.queues[role].pop(0)


def roles(generator):
    return ["rewrite" if r == REWRITE_PROMPT else "check" if r == CHECK_PROMPT else r for r, _ in generator.calls]


HITS = [hit("7-0", "231/25/2024-GST"), hit("9-1", "140/10/2020 - GST")]


def test_hindi_and_hinglish_questions_are_rewritten_english_ones_are_not():
    assert needs_rewrite("क्या डेमो गाड़ियों पर ITC मिलता है?")
    assert needs_rewrite("Gift voucher bechne par GST lagta hai kya?")
    assert not needs_rewrite("Is ITC available on demo vehicles?")


def test_checker_verdict_parsing():
    assert parse_unsupported('```json\n{"unsupported": ["A is B [1]."]}\n```') == ["A is B [1]."]
    assert parse_unsupported('{"unsupported": []}') == []
    assert parse_unsupported("Looks fine to me") is None


def test_supported_answer_is_checked_once_and_kept():
    generator = ScriptedGenerator(answers=[["ITC is allowed [1]."]], verdicts=[['{"unsupported": []}']])
    events = list(AnswerFlow(FakeRetriever(HITS), generator).stream("Is ITC available on demo vehicles?"))
    assert [e for e, _ in events] == ["sources", "token", "done"]
    assert events[-1] == ("done", [1])
    assert roles(generator) == ["answer", "check"]


def test_unsupported_answer_is_regenerated_once_with_feedback():
    generator = ScriptedGenerator(answers=[["ITC is always allowed [2]."], ["ITC is allowed on demo vehicles [1]."]],
                                  verdicts=[['{"unsupported": ["ITC is always allowed [2]."]}']])
    events = list(AnswerFlow(FakeRetriever(HITS), generator).stream("Is ITC available on demo vehicles?"))
    assert [e for e, _ in events] == ["sources", "token", "retry", "token", "done"]
    assert events[2] == ("retry", ["ITC is always allowed [2]."])
    assert events[-1] == ("done", [1])  # citations of the regenerated answer, not the first draft
    assert roles(generator) == ["answer", "check", "answer"]  # the final answer isn't checked again
    assert "ITC is always allowed [2]." in generator.calls[2][1]


def test_hinglish_question_searches_original_and_rewrite():
    retriever = FakeRetriever(HITS)
    generator = ScriptedGenerator(answers=[["Haan [1]."]], verdicts=[['{"unsupported": []}']],
                                  rewrites=[["input tax credit on demo vehicles\n"]])
    state = AnswerFlow(retriever, generator).run("Demo gaadi par ITC milta hai kya?")
    assert retriever.queries == ["Demo gaadi par ITC milta hai kya?", "input tax credit on demo vehicles"]
    assert state["query"] == "input tax credit on demo vehicles"
    assert state["answer"] == "Haan [1]."


def test_no_passages_means_no_answer_and_no_check():
    generator = ScriptedGenerator(answers=[])
    events = list(AnswerFlow(FakeRetriever([]), generator).stream("What is the capital of France?"))
    assert events == [("sources", []), ("done", [])]
    assert generator.calls == []


def test_api_forwards_retry_event():
    generator = ScriptedGenerator(answers=[["Wrong [2]."], ["Right [1]."]], verdicts=[['{"unsupported": ["Wrong [2]."]}']])
    api.app.dependency_overrides[api.get_retriever] = lambda: FakeRetriever(HITS)
    api.app.dependency_overrides[api.get_generator] = lambda: generator
    try:
        text = TestClient(api.app).post("/ask", json={"question": "Is ITC available on demo vehicles?"}).text
    finally:
        api.app.dependency_overrides.clear()
    blocks = [b.split("\n", 1) for b in text.strip().split("\n\n")]
    parsed = [(name.removeprefix("event: "), json.loads(data.removeprefix("data: "))) for name, data in blocks]
    assert [name for name, _ in parsed] == ["sources", "token", "retry", "token", "done"]
    assert parsed[-1][1] == {"cited": [1]}
