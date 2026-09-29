import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from lexeu.eval.answers import AnswerReport, check_gate, evaluate_answers, to_markdown
from lexeu.eval.golden import GoldenItem
from lexeu.eval.judge import Judge
from lexeu.generation.answer import Answerer
from lexeu.generation.conversation import Rewrite, Turn
from lexeu.generation.llm import Completion, Messages
from lexeu.retrieval.search import Hit

ART33 = "32016R0679:art_33:p1"
ART5 = "32016R0679:art_5"

HITS = [
    Hit(f"{ART33}:en", ART33, "en", "Art. 33(1) GDPR", 0.8, "72 hours", "article", ""),
    Hit(f"{ART5}:en", f"{ART5}:p1", "en", "Art. 5(1) GDPR", 0.7, "principles", "article", ""),
]

ITEMS = [
    GoldenItem(id="gdpr-001", lang="en", category="obligation", question="Breach deadline?",
               expected=[ART33], reference="72 hours (Art. 33(1))."),
    GoldenItem(id="gdpr-002", lang="en", category="lookup", question="What are the principles?",
               expected=[ART5], reference="Lawfulness, fairness..."),
    GoldenItem(id="none-001", lang="fr", category="unanswerable",
               question="Quelle TVA en Belgique ?", answerable=False, reference="Not covered."),
    GoldenItem(id="none-002", lang="en", category="unanswerable", question="Who won the 2022 cup?",
               answerable=False, reference="Not covered."),
]  # fmt: skip

# What the fake generator says for each question.
SCRIPT = {
    "Breach deadline?": {"kind": "answer", "answer": "Within 72 hours [1]."},
    "What are the principles?": {"kind": "not_in_sources", "answer": ""},  # false refusal
    "Quelle TVA en Belgique ?": {"kind": "not_in_sources", "answer": ""},  # correct refusal
    "Who won the 2022 cup?": {"kind": "answer", "answer": "See Article 5 [2]."},  # hallucination
}  # fmt: skip


class Retriever:
    async def search(self, query: str, k: int = 10, lang: str | None = None) -> list[Hit]:
        return HITS


class ScriptedLlm:
    model_id = "fake/gen"

    async def complete(self, messages: Messages, schema: type[BaseModel]) -> Completion:
        question = messages[-1]["content"].split("Question: ")[-1]
        return Completion(json.dumps(SCRIPT[question]), self.model_id, 800, 40, 0.002, 1000.0)


class JudgeLlm:
    model_id = "fake/judge"

    async def complete(self, messages: Messages, schema: type[BaseModel]) -> Completion:
        verdict = {"correctness": "correct", "faithful": True, "reason": "matches"}
        return Completion(json.dumps(verdict), self.model_id, 900, 30, 0.01, 2000.0)


async def _report() -> AnswerReport:
    answerer = Answerer(Retriever(), ScriptedLlm(), k=8)  # type: ignore[arg-type]
    return await evaluate_answers(ITEMS, answerer, Judge(JudgeLlm()), concurrency=2)


async def test_refusal_and_citation_metrics() -> None:
    report = await _report()
    o = report.overall
    assert o["refusal_recall"] == 0.5  # one of two unanswerable questions refused
    assert o["false_refusal_rate"] == 0.5  # one of two answerable questions refused
    assert o["citation_hit"] == 1.0  # the only answered answerable item cites Art. 33(1)
    assert o["retrieval_hit"] == 1.0
    assert o["correct"] == 1.0  # the judge only grades answered, answerable items
    assert report.cost["judge_total_usd"] == 0.02  # judged: gdpr-001 and the hallucination


async def test_markdown_lists_failures() -> None:
    md = to_markdown(await _report())
    assert "Answered although not answerable (1)" in md
    assert "`none-002`" in md
    assert "Refused although answerable (1)" in md


async def test_gate_supports_minimums_and_maximums(tmp_path: Path) -> None:
    report = await _report()
    path = tmp_path / "t.yaml"
    path.write_text(
        "answers:\n  refusal_recall: 0.9\n  false_refusal_rate_max: 0.6\n"
        "  latency_total_p95_max: 5000\n",
        encoding="utf-8",
    )
    assert check_gate(report, path) == ["refusal_recall = 0.5 < 0.9"]


class FlakyAnswerer:
    """Fails on one question, like a provider timeout."""

    def __init__(self) -> None:
        self.inner = Answerer(Retriever(), ScriptedLlm(), k=8)  # type: ignore[arg-type]

    async def answer(self, question: str, lang: str | None = None, history: object = None) -> Any:
        if question == "What are the principles?":
            raise TimeoutError("provider timed out")
        return await self.inner.answer(question, lang)


class BrokenJudgeLlm(JudgeLlm):
    async def complete(self, messages: Messages, schema: type[BaseModel]) -> Completion:
        raise RuntimeError("429 quota exceeded")


async def test_one_failure_does_not_abort_the_run() -> None:
    report = await evaluate_answers(ITEMS, FlakyAnswerer(), Judge(BrokenJudgeLlm()))  # type: ignore[arg-type]
    assert report.n_items == 4
    assert report.overall["error_rate"] == 0.25
    assert report.overall["n"] == 3  # the failed item is excluded from quality metrics
    assert report.overall["judge_error_rate"] == 1.0
    assert "correct" not in report.overall  # nothing could be judged
    assert "Errors (not scored)" in to_markdown(report)


# ----------------------------------------------------------------------------- conversations

CHAT_ITEMS = [
    GoldenItem(id="chat-001", lang="en", category="conversation", question="Hi!",
               answerable=False, reference="Greets and says what it covers."),
    GoldenItem(id="chat-002", lang="en", category="conversation", question="Who are you?",
               answerable=False, reference="Introduces itself."),
    GoldenItem(id="off-001", lang="en", category="out_of_scope", question="A crêpe recipe, please",
               answerable=False, reference="Out of scope."),
    GoldenItem(id="fu-001", lang="en", category="follow_up", question="And for processors?",
               expected=[ART33], reference="Art. 33.",
               history=[Turn(role="user", content="Breach deadline?"),
                        Turn(role="assistant", content="72 hours [1].")]),
]  # fmt: skip

CHAT_SCRIPT = {
    "Hi!": {"kind": "conversation", "answer": "Hello! Ask me about EU digital regulation."},
    "Who are you?": {"kind": "out_of_scope", "answer": ""},  # small talk not recognised
    "A crêpe recipe, please": {"kind": "out_of_scope", "answer": ""},
    "What is the breach deadline for processors?": {"kind": "answer", "answer": "Art. 33 [1]."},
}


class ChatLlm:
    model_id = "fake/gen"

    async def complete(self, messages: Messages, schema: type[BaseModel]) -> Completion:
        if schema is Rewrite:
            rewritten = "What is the breach deadline for processors?"
            return Completion(json.dumps({"question": rewritten}), self.model_id, 1, 1, 0.0, 1.0)
        question = messages[-1]["content"].split("Question: ")[-1]
        return Completion(json.dumps(CHAT_SCRIPT[question]), self.model_id, 1, 1, 0.0, 1.0)


async def test_small_talk_off_topic_and_follow_up_metrics() -> None:
    answerer = Answerer(Retriever(), ChatLlm(), k=8)  # type: ignore[arg-type]
    report = await evaluate_answers(CHAT_ITEMS, answerer, judge=None, concurrency=2)
    o = report.overall
    assert o["conversation_rate"] == 0.5  # "Who are you?" was taken for an off-topic request
    assert o["out_of_scope_recall"] == 1.0
    assert o["refusal_recall"] == 1.0  # small talk is not counted as a question to refuse
    follow_up = next(i for i in report.items if i.id == "fu-001")
    assert follow_up.standalone_question == "What is the breach deadline for processors?"
    assert follow_up.citation_hit is True
    md = to_markdown(report)
    assert "Small talk not recognised (1)" in md and "`chat-002`" in md
