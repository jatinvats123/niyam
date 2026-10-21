import asyncio

from niyam import mcp_server
from tests.test_graph import CHECK_PROMPT, HITS, FakeRetriever, ScriptedGenerator


def call(name, arguments):
    result = asyncio.run(mcp_server.server.call_tool(name, arguments))
    assert not result.is_error, result
    return result.structured_content


def test_tools_are_listed():
    tools = asyncio.run(mcp_server.server.list_tools())
    assert {t.name for t in tools} == {"search_circulars", "ask"}


def test_search_circulars_returns_citable_passages(monkeypatch):
    monkeypatch.setattr(mcp_server, "get_retriever", lambda: FakeRetriever(HITS))
    result = call("search_circulars", {"query": "ITC on demo cars", "k": 1})
    assert result["result"] == [{"n": 1, "circular": "231/25/2024-GST", "date": "2024-09-11",
                                 "subject": "ITC on demo vehicles", "pages": "page 2", "text": "Demo vehicles ..."}]


def test_ask_returns_answer_with_cited_sources(monkeypatch):
    generator = ScriptedGenerator(answers=[["ITC is allowed [1]."]], verdicts=[['{"unsupported": []}']])
    monkeypatch.setattr(mcp_server, "get_retriever", lambda: FakeRetriever(HITS))
    monkeypatch.setattr(mcp_server, "get_generator", lambda: generator)
    result = call("ask", {"question": "Is ITC available on demo vehicles?"})
    assert result["answer"] == "ITC is allowed [1]." and result["cited"] == [1]
    assert [s["circular"] for s in result["sources"]] == ["231/25/2024-GST", "140/10/2020 - GST"]
