"""MCP server: lets Claude Desktop, Cursor or any MCP client search the circulars and ask questions.

Run (stdio): python -m niyam.mcp_server
Tools: search_circulars(query, k) needs no LLM key; ask(question) needs GEMINI_API_KEY in .env.
"""

from __future__ import annotations

from typing import TypedDict

from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

from niyam.answer import cited_sources, page_label
from niyam.api import NOT_FOUND, ROOT, get_generator, get_retriever
from niyam.graph import AnswerFlow
from niyam.search import Hit

server = MCPServer(
    "niyam",
    instructions="Search official CBIC GST circulars (India) and get cited answers. Cite circular numbers "
                 "and pages from the results when you use them.",
)


class Passage(TypedDict):
    n: int
    circular: str
    date: str
    subject: str
    pages: str
    text: str


class Answer(TypedDict):
    answer: str
    cited: list[int]
    sources: list[Passage]


def source(n: int, hit: Hit) -> Passage:
    c = hit.chunk
    return {"n": n, "circular": c.number, "date": c.date, "subject": c.subject, "pages": page_label(hit),
            "text": c.text}


@server.tool()
def search_circulars(query: str, k: int = 6) -> list[Passage]:
    """Search the 272 CBIC GST circulars. Works with English, Hindi or Hinglish queries.

    Returns the k best passages (1-10), each with its circular number, date, subject, pages and text.
    """
    return [source(n, h) for n, h in enumerate(get_retriever().search(query, max(1, min(k, 10))), 1)]


@server.tool()
def ask(question: str) -> Answer:
    """Answer a GST question from the CBIC circulars, in the question's language, citing passages as [n].

    Returns the answer, the passage numbers it cites, and the passages. The answer is self-checked
    against its sources and regenerated once if a sentence isn't supported.
    """
    state = AnswerFlow(get_retriever(), get_generator()).run(question)
    hits = state.get("hits", [])
    if not hits:
        return {"answer": NOT_FOUND, "cited": [], "sources": []}
    return {"answer": state["answer"], "cited": cited_sources(state["answer"], len(hits)),
            "sources": [source(n, h) for n, h in enumerate(hits, 1)]}


def main() -> None:
    load_dotenv(ROOT / ".env")
    server.run()  # stdio


if __name__ == "__main__":
    main()
