"""The answer flow as a LangGraph graph: rewrite the query -> retrieve -> answer -> self-check.

    rewrite ──> retrieve ──> answer ──> check ──(unsupported, first try)──> answer
                                          └───(supported, or already retried)──> END

Nodes report progress through LangGraph's custom stream, so the API can forward passages and answer
tokens to the client as they are produced. `AnswerFlow.stream()` yields (event, payload) pairs:
("sources", hits), ("token", text), ("retry", unsupported sentences), ("done", cited passage numbers).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Protocol, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from niyam.answer import SYSTEM_PROMPT, Generator, build_prompt, cited_sources
from niyam.search import Hit, reciprocal_rank_fusion

REWRITE_PROMPT = """Rewrite the user's GST question as a short English search query for CBIC GST circulars.
Translate Hindi or Hinglish into English, use the official terms (input tax credit, place of supply,
e-invoice, GSTR-9C, late fee), keep form names and section numbers, and drop filler words.
Reply with the search query only, on one line."""

CHECK_PROMPT = """You check a cited answer against its numbered source passages.
For each sentence of the ANSWER that states a fact, decide whether the passage(s) it cites in square
brackets actually support it. A factual sentence with no citation is unsupported. Reply with JSON only:
{"unsupported": ["<sentence>", ...]}  (an empty list when every sentence is supported)"""

RETRY_NOTE = """Your previous answer had sentences that the cited passages do not support:
{sentences}
Answer again. Use only what the passages state, cite each sentence correctly, and leave out anything
they do not support."""

# Words that mark a question as Hinglish (Hindi written in Latin script). English questions skip the
# rewrite call: e5 already retrieves them well, so the extra LLM round trip would only add latency.
HINGLISH = re.compile(r"\b(kya|hai|hain|ka|ki|ke|ko|kaise|kab|kaun|kitna|kitni|mein|par|pe|nahi|nahin|"
                      r"lagta|lagti|lagega|milta|milega|karna|karne|chahiye|bharega|bechne|wala|wali)\b",
                      re.IGNORECASE)


class Searcher(Protocol):
    def search(self, query: str, k: int = 6) -> list[Hit]: ...


class State(TypedDict, total=False):
    question: str
    k: int
    query: str  # what was searched: the question itself, or its English rewrite
    hits: list[Hit]
    answer: str
    unsupported: list[str]
    attempts: int


def needs_rewrite(question: str) -> bool:
    return not question.isascii() or bool(HINGLISH.search(question))


def parse_unsupported(verdict: str) -> list[str] | None:
    """The sentences the checker flagged, or None if its reply isn't the JSON asked for."""
    match = re.search(r"\{.*\}", verdict, re.DOTALL)
    try:
        flagged = json.loads(match.group(0))["unsupported"] if match else None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return [str(s) for s in flagged] if isinstance(flagged, list) else None


class AnswerFlow:
    def __init__(self, retriever: Searcher, generator: Generator):
        self.retriever = retriever
        self.generator = generator
        graph = StateGraph(State)
        graph.add_node("rewrite", self.rewrite)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("answer", self.answer)
        graph.add_node("check", self.check)
        graph.add_edge(START, "rewrite")
        graph.add_edge("rewrite", "retrieve")
        graph.add_conditional_edges("retrieve", lambda s: "answer" if s["hits"] else END)
        graph.add_edge("answer", "check")
        graph.add_conditional_edges("check", lambda s: "answer" if s.get("unsupported") and s["attempts"] < 2 else END)
        self.graph = graph.compile()

    def complete(self, system: str, prompt: str) -> str:
        return "".join(self.generator.stream(system, prompt)).strip()

    def rewrite(self, state: State) -> State:
        question = state["question"]
        query = self.complete(REWRITE_PROMPT, question).splitlines()[0] if needs_rewrite(question) else question
        return {"query": query or question}

    def retrieve(self, state: State) -> State:
        question, query, k = state["question"], state["query"], state["k"]
        if query == question:
            hits = self.retriever.search(question, k)
        else:  # search both: the rewrite fixes vocabulary, the original keeps anything the rewrite lost
            hits = reciprocal_rank_fusion(self.retriever.search(question, 20), self.retriever.search(query, 20), k=k)
        get_stream_writer()(("sources", hits))
        return {"hits": hits, "attempts": 0}

    def answer(self, state: State) -> State:
        write = get_stream_writer()
        prompt = build_prompt(state["question"], state["hits"])
        if state.get("unsupported"):
            write(("retry", state["unsupported"]))
            prompt += "\n\n" + RETRY_NOTE.format(sentences="\n".join(f"- {s}" for s in state["unsupported"]))
        parts = []
        for text in self.generator.stream(SYSTEM_PROMPT, prompt):
            parts.append(text)
            write(("token", text))
        return {"answer": "".join(parts), "attempts": state["attempts"] + 1, "unsupported": []}

    def check(self, state: State) -> State:
        if state["attempts"] >= 2:  # the regenerated answer is final; don't pay for a check we won't act on
            return {}
        prompt = f"{build_prompt(state['question'], state['hits'])}\n\nANSWER:\n{state['answer']}"
        flagged = parse_unsupported(self.complete(CHECK_PROMPT, prompt))
        # A checker reply we can't read is not evidence against the answer: keep it rather than fail.
        return {"unsupported": flagged or []}

    def stream(self, question: str, k: int = 6) -> Iterator[tuple[str, object]]:
        final: State = {}
        for mode, chunk in self.graph.stream({"question": question, "k": k}, stream_mode=["custom", "values"]):
            if mode == "custom":
                yield chunk
            else:
                final = chunk
        yield "done", cited_sources(final.get("answer", ""), len(final.get("hits", [])))

    def run(self, question: str, k: int = 6) -> State:
        """Run the whole flow without streaming; returns the final state (used by the eval and MCP)."""
        return self.graph.invoke({"question": question, "k": k})
