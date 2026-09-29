"""Run the golden set against a retriever and aggregate metrics.

Metrics are computed on items that have expected provisions. Unanswerable items have no
retrieval target, but their top score is recorded: comparing it with the top score of
answerable items tells us whether a score threshold can drive refusals (used in M4).
"""

import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from lexeu.eval.golden import GoldenItem
from lexeu.eval.metrics import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from lexeu.retrieval.search import Hit

KS = (1, 3, 5, 10)


class Retriever(Protocol):
    async def search(self, query: str, k: int = 10, lang: str | None = None) -> list[Hit]: ...


@dataclass(frozen=True)
class ItemResult:
    id: str
    lang: str
    category: str
    answerable: bool
    expected: list[str]
    retrieved: list[str]
    citations: list[str]
    top_score: float
    latency_ms: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalReport:
    k: int
    n_items: int
    overall: dict[str, float]
    by_category: dict[str, dict[str, float]]
    by_lang: dict[str, dict[str, float]]
    scores: dict[str, float]
    latency_ms: dict[str, float]
    items: list[ItemResult]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


async def evaluate_retrieval(
    items: Iterable[GoldenItem],
    retriever: Retriever,
    k: int = 10,
    on_item: Callable[[ItemResult], None] | None = None,
) -> RetrievalReport:
    results: list[ItemResult] = []
    for item in items:
        start = time.perf_counter()
        hits = await retriever.search(item.question, k=k)
        latency = (time.perf_counter() - start) * 1000
        retrieved = [h.provision_key for h in hits]
        metrics = _item_metrics(retrieved, item.expected, k) if item.expected else {}
        result = ItemResult(
            id=item.id,
            lang=item.lang,
            category=item.category,
            answerable=item.answerable,
            expected=item.expected,
            retrieved=retrieved,
            citations=[h.citation for h in hits],
            top_score=hits[0].score if hits else 0.0,
            latency_ms=round(latency, 1),
            metrics=metrics,
        )
        results.append(result)
        if on_item:
            on_item(result)

    scored = [r for r in results if r.metrics]
    return RetrievalReport(
        k=k,
        n_items=len(results),
        overall=_mean_metrics(scored),
        by_category=_grouped(scored, lambda r: r.category),
        by_lang=_grouped(scored, lambda r: r.lang),
        scores=_score_separation(results),
        latency_ms=_latency(results),
        items=results,
    )


def _item_metrics(retrieved: list[str], expected: list[str], k: int) -> dict[str, float]:
    m = {f"hit@{n}": hit_at_k(retrieved, expected, n) for n in KS if n <= k}
    m[f"recall@{k}"] = recall_at_k(retrieved, expected, k)
    m[f"mrr@{k}"] = mrr_at_k(retrieved, expected, k)
    m[f"ndcg@{k}"] = ndcg_at_k(retrieved, expected, k)
    return m


def _mean_metrics(results: list[ItemResult]) -> dict[str, float]:
    if not results:
        return {}
    names = results[0].metrics.keys()
    out = {n: round(statistics.fmean(r.metrics[n] for r in results), 4) for n in names}
    out["n"] = len(results)
    return out


def _grouped(
    results: list[ItemResult], key: Callable[[ItemResult], str]
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[ItemResult]] = defaultdict(list)
    for r in results:
        groups[key(r)].append(r)
    return {g: _mean_metrics(rs) for g, rs in sorted(groups.items())}


def _score_separation(results: list[ItemResult]) -> dict[str, float]:
    answerable = [r.top_score for r in results if r.answerable]
    unanswerable = [r.top_score for r in results if not r.answerable]
    out: dict[str, float] = {}
    if answerable:
        out["top_score_answerable_median"] = round(statistics.median(answerable), 4)
        out["top_score_answerable_p10"] = round(_quantile(answerable, 0.10), 4)
    if unanswerable:
        out["top_score_unanswerable_median"] = round(statistics.median(unanswerable), 4)
        out["top_score_unanswerable_p90"] = round(_quantile(unanswerable, 0.90), 4)
    return out


def _latency(results: list[ItemResult]) -> dict[str, float]:
    values = [r.latency_ms for r in results]
    return {"p50": round(_quantile(values, 0.5), 1), "p95": round(_quantile(values, 0.95), 1)}


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


# ------------------------------------------------------------------------------ reporting


def to_markdown(report: RetrievalReport, title: str = "Retrieval evaluation") -> str:
    metric_names = [m for m in report.overall if m != "n"]
    header = "| Slice | n | " + " | ".join(metric_names) + " |"
    sep = "|---|---|" + "---|" * len(metric_names)

    def row(name: str, m: dict[str, float]) -> str:
        return (
            f"| {name} | {int(m['n'])} | " + " | ".join(f"{m[n]:.3f}" for n in metric_names) + " |"
        )

    lines = [f"## {title}", "", header, sep, row("**overall**", report.overall)]
    lines += [row(f"category: {c}", m) for c, m in report.by_category.items()]
    lines += [row(f"lang: {lang}", m) for lang, m in report.by_lang.items()]
    lines += [
        "",
        f"Latency per query: p50 {report.latency_ms['p50']} ms, p95 {report.latency_ms['p95']} ms.",
        "",
        "Top-1 score, answerable vs unanswerable (can a threshold drive refusals?): "
        + ", ".join(f"{k} = {v}" for k, v in report.scores.items()),
    ]
    misses = [r for r in report.items if r.metrics and r.metrics.get(f"hit@{report.k}") == 0.0]
    if misses:
        lines += ["", f"### Misses at k={report.k} ({len(misses)})", ""]
        lines += [
            f"- `{r.id}` expected {', '.join(r.expected)}; got {', '.join(r.citations[:3])}"
            for r in misses
        ]
    return "\n".join(lines) + "\n"


def ablation_markdown(results: list[tuple[Any, RetrievalReport]], k: int) -> str:
    """One row per config, with the delta against the first one (the baseline)."""
    cols = ["hit@1", "hit@5", f"hit@{k}", f"mrr@{k}", f"ndcg@{k}"]
    slices = ["definition", "temporal", "cross_regulation"]
    base_cfg, base = results[0]
    header = (
        "| Config | "
        + " | ".join(cols)
        + " | "
        + " | ".join(f"hit@5 {s}" for s in slices)
        + " | p95 ms |"
    )
    lines = [f"## Retrieval ablation (k={k}, baseline = `{base_cfg.name}`)", "", header]
    lines.append("|---" * (len(cols) + len(slices) + 2) + "|")
    for cfg, rep in results:
        cells = []
        for c in cols:
            v, b = rep.overall.get(c, 0.0), base.overall.get(c, 0.0)
            cells.append(f"{v:.3f}" if cfg is base_cfg else f"{v:.3f} ({v - b:+.3f})")
        for sl in slices:
            cells.append(f"{rep.by_category.get(sl, {}).get('hit@5', 0.0):.3f}")
        lines.append(f"| `{cfg.name}` | " + " | ".join(cells) + f" | {rep.latency_ms['p95']} |")
    lines.append("")
    lines += [f"- `{cfg.name}`: {cfg.description}" for cfg, _ in results if cfg.description]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------ quality gate


def check_gate(report: RetrievalReport, thresholds_path: Path) -> list[str]:
    """Return the list of violated thresholds (empty = gate passed)."""
    spec = yaml.safe_load(thresholds_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for metric, minimum in spec.get("overall", {}).items():
        value = report.overall.get(metric)
        if value is None or value < minimum:
            failures.append(f"overall {metric} = {value} < {minimum}")
    for slice_name, mins in spec.get("by_category", {}).items():
        for metric, minimum in mins.items():
            value = report.by_category.get(slice_name, {}).get(metric)
            if value is None or value < minimum:
                failures.append(f"{slice_name} {metric} = {value} < {minimum}")
    return failures
