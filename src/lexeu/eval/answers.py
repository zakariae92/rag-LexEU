"""End-to-end evaluation: run the golden set through the answerer and grade the answers.

Deterministic metrics (primary, free to compute):
- refusal_recall: share of unanswerable questions the system refuses ("I don't know" works)
- false_refusal_rate: share of answerable questions it wrongly refuses (or treats as small talk)
- citation_hit: answered questions whose citations include an expected provision
- citation_precision: share of cited provisions that are expected ones (a lower bound: citing a
  relevant provision the golden set does not list counts against it)
- retrieval_hit: an expected provision was among the sources (separates retrieval from
  generation failures)
- invalid_citation_rate, uncited_sentence_ratio: grounding hygiene
- conversation_rate: small talk answered as small talk (not refused, not "answered" with law)
- out_of_scope_recall: off-topic requests refused as out of scope
Follow-ups are asked with their `history`: the whole path, rewrite included, is measured.
LLM judge (second signal): correct, correct_or_partial, faithful.
Operations: latency p50/p95 end to end, cost and tokens per question.
"""

import asyncio
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from lexeu.eval.golden import GoldenItem
from lexeu.eval.judge import Judge
from lexeu.eval.metrics import matches, quantile
from lexeu.generation.answer import Answer, Answerer


@dataclass(frozen=True)
class AnswerItem:
    id: str
    lang: str
    category: str
    answerable: bool
    question: str
    reference: str
    expected: list[str]
    answer: str
    refused: bool
    refusal_reason: str | None
    conversation: bool
    standalone_question: str | None
    cited: list[str]
    sources: list[str]
    retrieval_hit: bool | None
    citation_hit: bool | None
    citation_precision: float | None
    invalid_citations: int
    uncited_ratio: float
    correctness: str | None
    faithful: bool | None
    judge_reason: str | None
    latency_ms: float
    retrieval_ms: float
    generation_ms: float
    cost_usd: float
    input_tokens: int
    output_tokens: int
    judge_cost_usd: float
    cached: bool
    error: str | None = None  # the system failed (timeout, quota): excluded from quality metrics
    judge_error: str | None = None


@dataclass(frozen=True)
class AnswerReport:
    n_items: int
    model: str
    judge_model: str | None
    overall: dict[str, float]
    by_category: dict[str, dict[str, float]]
    by_lang: dict[str, dict[str, float]]
    latency_ms: dict[str, float]
    cost: dict[str, float]
    items: list[AnswerItem]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


async def evaluate_answers(
    items: Iterable[GoldenItem],
    answerer: Answerer,
    judge: Judge | None = None,
    concurrency: int = 4,
    on_item: Callable[[AnswerItem], None] | None = None,
) -> AnswerReport:
    sem = asyncio.Semaphore(concurrency)
    model = ""

    async def run(item: GoldenItem) -> AnswerItem:
        # One failed call (timeout, quota) must not throw away a whole run: it is recorded and
        # reported as an error rate, and the cached answers make the rerun cheap.
        nonlocal model
        async with sem:
            start = time.perf_counter()
            try:
                answer = await answerer.answer(
                    item.question, lang=item.lang, history=item.history or None
                )
            except Exception as exc:
                result = _error_item(item, exc, (time.perf_counter() - start) * 1000)
            else:
                model = answer.model
                verdict, judge_cost, judge_error = None, 0.0, None
                if judge is not None and not answer.refused and not answer.conversation:
                    try:
                        verdict, completion = await judge.grade(answer, item.reference)
                        judge_cost = completion.cost_usd
                    except Exception as exc:
                        judge_error = _describe(exc)
                result = _item(item, answer, verdict, judge_cost, judge_error)
        if on_item:
            on_item(result)
        return result

    results = list(await asyncio.gather(*(run(i) for i in items)))
    return AnswerReport(
        n_items=len(results),
        model=model,
        judge_model=judge.model_id if judge else None,
        overall=_metrics(results),
        by_category={k: _metrics(v) for k, v in _grouped(results, lambda r: r.category).items()},
        by_lang={k: _metrics(v) for k, v in _grouped(results, lambda r: r.lang).items()},
        latency_ms=_latency(results),
        cost=_cost(results),
        items=results,
    )


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:200]}"


def _error_item(item: GoldenItem, exc: BaseException, latency_ms: float) -> AnswerItem:
    return AnswerItem(
        id=item.id, lang=item.lang, category=item.category, answerable=item.answerable,
        question=item.question, reference=item.reference, expected=item.expected, answer="",
        refused=False, refusal_reason=None, conversation=False, standalone_question=None,
        cited=[], sources=[], retrieval_hit=None,
        citation_hit=None, citation_precision=None, invalid_citations=0, uncited_ratio=0.0,
        correctness=None, faithful=None, judge_reason=None, latency_ms=round(latency_ms, 1),
        retrieval_ms=0.0, generation_ms=0.0, cost_usd=0.0, input_tokens=0, output_tokens=0,
        judge_cost_usd=0.0, cached=False, error=_describe(exc),
    )  # fmt: skip


def _item(
    item: GoldenItem, a: Answer, verdict: Any, judge_cost: float, judge_error: str | None = None
) -> AnswerItem:
    cited = [c.provision_key for c in a.citations]
    sources = [h.provision_key for h in a.sources]
    scored = item.answerable and bool(item.expected) and not a.refused and bool(cited)
    relevant = [c for c in cited if any(matches(c, e) for e in item.expected)]
    return AnswerItem(
        id=item.id,
        lang=item.lang,
        category=item.category,
        answerable=item.answerable,
        question=item.question,
        reference=item.reference,
        expected=item.expected,
        answer=a.text,
        refused=a.refused,
        refusal_reason=a.refusal_reason,
        conversation=a.conversation,
        standalone_question=a.standalone_question,
        cited=cited,
        sources=sources,
        retrieval_hit=(
            any(matches(s, e) for s in sources for e in item.expected) if item.expected else None
        ),
        citation_hit=bool(relevant) if scored else None,
        citation_precision=round(len(relevant) / len(cited), 3) if scored else None,
        invalid_citations=len(a.invalid_citations),
        uncited_ratio=a.uncited_ratio,
        correctness=verdict.correctness if verdict else None,
        faithful=verdict.faithful if verdict else None,
        judge_reason=verdict.reason if verdict else None,
        latency_ms=a.timings_ms.get("total", a.timings_ms.get("retrieval", 0.0)),
        retrieval_ms=a.timings_ms.get("retrieval", 0.0),
        generation_ms=a.timings_ms.get("generation", 0.0),
        cost_usd=a.cost_usd,
        input_tokens=a.input_tokens,
        output_tokens=a.output_tokens,
        judge_cost_usd=judge_cost,
        cached=a.cached,
        judge_error=judge_error,
    )


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _metrics(all_results: list[AnswerItem]) -> dict[str, float]:
    results = [r for r in all_results if r.error is None]
    # Small talk is neither a question to refuse nor one to answer from the law.
    unanswerable = [r for r in results if not r.answerable and r.category != "conversation"]
    answerable = [r for r in results if r.answerable]
    answered = [r for r in results if not r.refused and not r.conversation]
    judged = [r for r in answered if r.answerable and r.correctness is not None]
    candidates: dict[str, float | None] = {
        "n": len(results),
        "error_rate": _rate([r.error is not None for r in all_results]),
        "judge_error_rate": _rate([r.judge_error is not None for r in results if not r.refused]),
        "refusal_recall": _rate([r.refused for r in unanswerable]),
        "false_refusal_rate": _rate([r.refused or r.conversation for r in answerable]),
        "conversation_rate": _rate(
            [r.conversation for r in results if r.category == "conversation"]
        ),
        "out_of_scope_recall": _rate(
            [r.refusal_reason == "out_of_scope" for r in results if r.category == "out_of_scope"]
        ),
        "retrieval_hit": _rate([r.retrieval_hit for r in results if r.retrieval_hit is not None]),
        "citation_hit": _rate([r.citation_hit for r in results if r.citation_hit is not None]),
        "citation_precision": (
            round(statistics.mean(ps), 4)
            if (ps := [r.citation_precision for r in results if r.citation_precision is not None])
            else None
        ),
        "invalid_citation_rate": _rate([r.invalid_citations > 0 for r in answered]),
        "uncited_sentence_ratio": (
            round(statistics.mean(r.uncited_ratio for r in answered), 4) if answered else None
        ),
        "correct": _rate([r.correctness == "correct" for r in judged]),
        "correct_or_partial": _rate([r.correctness != "incorrect" for r in judged]),
        "faithful": _rate([bool(r.faithful) for r in judged]),
    }
    return {k: v for k, v in candidates.items() if v is not None}


def _grouped(
    results: list[AnswerItem], key: Callable[[AnswerItem], str]
) -> dict[str, list[AnswerItem]]:
    groups: dict[str, list[AnswerItem]] = defaultdict(list)
    for r in results:
        groups[key(r)].append(r)
    return dict(sorted(groups.items()))


def _latency(all_results: list[AnswerItem]) -> dict[str, float]:
    results = [r for r in all_results if r.error is None]
    out: dict[str, float] = {}
    for name, values in (
        ("total", [r.latency_ms for r in results]),
        ("retrieval", [r.retrieval_ms for r in results]),
        ("generation", [r.generation_ms for r in results if r.generation_ms]),
    ):
        out[f"{name}_p50"] = round(quantile(values, 0.5), 1)
        out[f"{name}_p95"] = round(quantile(values, 0.95), 1)
    return out


def _cost(results: list[AnswerItem]) -> dict[str, float]:
    n = len(results) or 1
    return {
        "per_question_usd": round(sum(r.cost_usd for r in results) / n, 6),
        "per_1k_questions_usd": round(sum(r.cost_usd for r in results) / n * 1000, 2),
        "total_usd": round(sum(r.cost_usd for r in results), 4),
        "judge_total_usd": round(sum(r.judge_cost_usd for r in results), 4),
        "input_tokens_mean": round(sum(r.input_tokens for r in results) / n, 1),
        "output_tokens_mean": round(sum(r.output_tokens for r in results) / n, 1),
        "cache_hit_rate": round(sum(r.cached for r in results) / n, 4),
    }


# ------------------------------------------------------------------------------ reporting

_COLUMNS = [
    ("refusal_recall", "refuse (unanswerable)"),
    ("false_refusal_rate", "false refusal"),
    ("conversation_rate", "small talk"),
    ("out_of_scope_recall", "off-topic refused"),
    ("retrieval_hit", "retrieval hit"),
    ("citation_hit", "citation hit"),
    ("citation_precision", "citation precision"),
    ("correct", "correct"),
    ("correct_or_partial", "correct+partial"),
    ("faithful", "faithful"),
]


def to_markdown(report: AnswerReport, title: str = "Answer evaluation") -> str:
    def row(name: str, m: dict[str, float]) -> str:
        cells = [f"{m[c]:.3f}" if c in m else "-" for c, _ in _COLUMNS]
        return f"| {name} | {int(m['n'])} | " + " | ".join(cells) + " |"

    judge = f", judge `{report.judge_model}`" if report.judge_model else ""
    lines = [
        f"## {title}",
        "",
        f"Model `{report.model}`{judge}, {report.n_items} questions.",
        "",
        "| Slice | n | " + " | ".join(label for _, label in _COLUMNS) + " |",
        "|---|---|" + "---|" * len(_COLUMNS),
        row("**overall**", report.overall),
    ]
    lines += [row(f"category: {k}", v) for k, v in report.by_category.items()]
    lines += [row(f"lang: {k}", v) for k, v in report.by_lang.items()]
    lat, cost = report.latency_ms, report.cost
    lines += [
        "",
        f"Latency: end to end p50 {lat['total_p50']:.0f} ms, **p95 {lat['total_p95']:.0f} ms** "
        f"(retrieval p95 {lat['retrieval_p95']:.0f} ms, generation p95 "
        f"{lat['generation_p95']:.0f} ms).",
        f"Cost: **${cost['per_question_usd']:.5f} per question** "
        f"(${cost['per_1k_questions_usd']:.2f} per 1,000), {cost['input_tokens_mean']:.0f} input + "
        f"{cost['output_tokens_mean']:.0f} output tokens on average. Judge: "
        f"${cost['judge_total_usd']:.4f} total. Response cache hits: {cost['cache_hit_rate']:.0%}.",
    ]
    lines += _failures(report.items)
    return "\n".join(lines) + "\n"


def _failures(items: list[AnswerItem]) -> list[str]:
    sections = [
        (
            "Answered although not answerable",
            [i for i in items if not (i.answerable or i.refused) and i.category != "conversation"],
        ),
        (
            "Small talk not recognised",
            [i for i in items if i.category == "conversation" and not i.conversation],
        ),
        (
            "Off-topic not refused as such",
            [
                i
                for i in items
                if i.category == "out_of_scope" and i.refusal_reason != "out_of_scope"
            ],
        ),
        (
            "Legal question treated as small talk",
            [i for i in items if i.answerable and i.conversation],
        ),
        ("Refused although answerable", [i for i in items if i.answerable and i.refused]),
        ("Judged incorrect", [i for i in items if i.correctness == "incorrect"]),
        ("Judged unfaithful", [i for i in items if i.faithful is False]),
        ("Errors (not scored)", [i for i in items if i.error or i.judge_error]),
    ]
    out: list[str] = []
    for title, rows in sections:
        if not rows:
            continue
        out += ["", f"### {title} ({len(rows)})", ""]
        for i in rows:
            detail = i.judge_reason or i.refusal_reason or i.answer[:160].replace("\n", " ")
            if i.standalone_question:
                detail = f"(understood as: {i.standalone_question}) {detail}"
            hit = "" if i.retrieval_hit is None else f" (source retrieved: {i.retrieval_hit})"
            out.append(f"- `{i.id}` {i.question}{hit}: {detail}")
    return out


def check_gate(report: AnswerReport, thresholds_path: Path) -> list[str]:
    """`answers:` section of the thresholds file. Keys ending in `_max` are upper bounds."""
    spec = (yaml.safe_load(thresholds_path.read_text(encoding="utf-8")) or {}).get("answers", {})
    values = {**report.overall, **{f"latency_{k}": v for k, v in report.latency_ms.items()}}
    values.update({f"cost_{k}": v for k, v in report.cost.items()})
    failures: list[str] = []
    for metric, bound in spec.items():
        name, is_max = (metric[:-4], True) if metric.endswith("_max") else (metric, False)
        value = values.get(name)
        if value is None or (value > bound if is_max else value < bound):
            failures.append(f"{name} = {value} {'>' if is_max else '<'} {bound}")
    return failures
