"""End-to-end answer eval: retrieval + generation on the question set. Needs GEMINI_API_KEY.

Per question:
- answerable: does the answer cite at least one passage, and does a citation come from a relevant
  circular? Citation precision = share of cited passages that come from relevant circulars.
- out-of-scope (no relevant circular): does the answer decline instead of inventing one?
- with --judge: an LLM judge counts how many of the answer's claims the cited passages support.

--flow graph (default) runs the LangGraph flow the API uses (rewrite, retrieve, answer, self-check);
--flow plain is retrieve + answer only, to measure what the rewrite and self-check add.

Usage: python scripts/eval_answers.py [--judge] [--flow graph|plain] [--limit N] [--gate]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from niyam.answer import SYSTEM_PROMPT, GeminiGenerator, build_prompt, cited_sources
from niyam.graph import AnswerFlow
from niyam.search import Retriever

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data" / "eval" / "questions.jsonl"
THRESHOLDS = {"cites_relevant": 0.85, "declines_out_of_scope": 0.8, "faithfulness": 0.9}
DECLINE = re.compile(r"could(?:n't| not) find|not (?:found|covered)|नहीं मिल|nahi mil", re.IGNORECASE)

JUDGE_PROMPT = """You check whether an answer is supported by its sources.
Split the ANSWER into its factual claims. For each claim decide whether the numbered PASSAGES it cites
(or, if it cites none, any passage) state it. Reply with JSON only:
{{"claims": <int>, "supported": <int>, "unsupported": ["<claim>", ...]}}

PASSAGES:
{passages}

QUESTION: {question}

ANSWER:
{answer}"""


def with_retry(fn, attempts: int = 5):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # free-tier rate limits (429) and transient 5xx
            if attempt == attempts - 1 or not re.search(r"429|RESOURCE_EXHAUSTED|503|UNAVAILABLE", str(exc)):
                raise
            time.sleep(10 * (attempt + 1))


def judge(generator: GeminiGenerator, question: str, passages: str, answer: str) -> float:
    from google.genai import types

    response = with_retry(lambda: generator.client.models.generate_content(
        model=generator.model,
        contents=JUDGE_PROMPT.format(passages=passages, question=question, answer=answer),
        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
    ))
    verdict = json.loads(response.text)
    return verdict["supported"] / verdict["claims"] if verdict.get("claims") else 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true", help="also score faithfulness with an LLM judge")
    parser.add_argument("--flow", choices=("graph", "plain"), default="graph")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gate", action="store_true", help=f"exit 1 below thresholds {THRESHOLDS}")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")

    questions = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = questions[: args.limit]
    retriever, generator = Retriever(), GeminiGenerator()
    flow = AnswerFlow(retriever, generator)

    cites_relevant, precision, declines, faithfulness = [], [], [], []
    retried = 0
    for q in questions:
        if args.flow == "graph":
            state = with_retry(lambda: flow.run(q["q"]))
            hits, answer = state.get("hits", []), state.get("answer", "")
            retried += bool(state.get("attempts", 0) > 1)
        else:
            hits = retriever.search(q["q"], 6)
            answer = with_retry(lambda: "".join(generator.stream(SYSTEM_PROMPT, build_prompt(q["q"], hits))))
        prompt = build_prompt(q["q"], hits)
        cited = cited_sources(answer, len(hits))
        relevant = set(q["relevant"])
        if relevant:
            from_relevant = [n for n in cited if hits[n - 1].chunk.number in relevant]
            cites_relevant.append(bool(from_relevant))
            precision.append(len(from_relevant) / len(cited) if cited else 0.0)
            status = f"cited {cited}, relevant {from_relevant}"
        else:
            declined = not cited and (not hits or bool(DECLINE.search(answer)))
            declines.append(declined)
            status = "declined" if declined else "ANSWERED AN OUT-OF-SCOPE QUESTION"
        if args.judge and cited:
            faithfulness.append(judge(generator, q["q"], prompt, answer))
            status += f", faithfulness {faithfulness[-1]:.2f}"
        print(f"{q['id']} [{q['lang']}] {status}", flush=True)

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    summary = {"cites_relevant": mean(cites_relevant), "citation_precision": mean(precision),
               "declines_out_of_scope": mean(declines)}
    if faithfulness:
        summary["faithfulness"] = mean(faithfulness)
    print("\n" + "  ".join(f"{k} {v:.2f}" for k, v in summary.items()))

    if args.gate:
        failures = [f"{k} {summary[k]:.2f} < {t}" for k, t in THRESHOLDS.items() if k in summary and summary[k] < t]
        for failure in failures:
            print(f"GATE FAILED: {failure}")
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
