"""`lexeu eval validate | retrieval | ablation | answers | cache | review`."""

import asyncio
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from sqlalchemy import select

from lexeu.commands.ask import answerer_session
from lexeu.commands.index import make_embedder
from lexeu.core.config import Settings, get_settings
from lexeu.core.logging import configure_logging
from lexeu.eval.answers import AnswerItem, AnswerReport, evaluate_answers
from lexeu.eval.answers import check_gate as check_answers_gate
from lexeu.eval.answers import to_markdown as answers_markdown
from lexeu.eval.cache import check_lang_consistency, evaluate_cache, load_pairs
from lexeu.eval.cache import to_markdown as cache_markdown
from lexeu.eval.golden import (
    GoldenSet,
    Review,
    load_golden,
    load_reviews,
    save_reviews,
    status_of,
)
from lexeu.eval.judge import JUDGE_VERSION, Judge
from lexeu.eval.metrics import matches
from lexeu.eval.retrieval import (
    RetrievalReport,
    ablation_markdown,
    check_gate,
    evaluate_retrieval,
    to_markdown,
)
from lexeu.generation.factory import make_llm
from lexeu.generation.prompt import PROMPT_VERSION
from lexeu.infra.db import ChunkRow, IndexRecord, SqlIndexRegistry
from lexeu.infra.resources import make_engine, make_qdrant
from lexeu.retrieval.config import RetrievalConfig, load_experiments
from lexeu.retrieval.index import ensure_same_model
from lexeu.retrieval.rerank import make_reranker, remote_rerankers
from lexeu.retrieval.search import Retriever

app = typer.Typer(help="Evaluate the system against the golden set.", no_args_is_help=True)
console = Console()

GoldenOpt = Annotated[Path | None, typer.Option("--golden", help="Golden set YAML.")]


@app.command()
def validate(golden: GoldenOpt = None) -> None:
    """Check that every expected provision exists in the ingested corpus."""
    asyncio.run(_validate(golden))


async def _validate(golden: Path | None) -> None:
    settings = get_settings()
    path = golden or Path(settings.eval.golden_path)
    gs = load_golden(path)
    engine = make_engine(settings)
    try:
        async with engine.connect() as conn:
            keys = set((await conn.execute(select(ChunkRow.provision_key).distinct())).scalars())
    finally:
        await engine.dispose()

    missing = [
        (item.id, key)
        for item in gs.items
        for key in item.expected
        if not any(matches(k, key) for k in keys)
    ]
    reviews = load_reviews(path)
    counts: dict[str, int] = {}
    for item in gs.items:
        s = status_of(item, reviews)
        counts[s] = counts.get(s, 0) + 1
    console.print(f"{len(gs.items)} items, review status: {counts}")
    if missing:
        for item_id, key in missing:
            console.print(f"[red]{item_id}: {key} not found in the corpus[/]")
        raise typer.Exit(1)
    console.print("[green]All expected provisions exist in the corpus.[/]")


ExperimentOpt = Annotated[
    str | None, typer.Option("--experiment", "-e", help="Config name in the experiments file.")
]


@app.command()
def retrieval(
    golden: GoldenOpt = None,
    experiment: ExperimentOpt = None,
    k: Annotated[int, typer.Option(help="Cut-off for metrics.")] = 10,
    gate: Annotated[bool, typer.Option(help="Fail if below eval/thresholds.yaml.")] = False,
    mlflow: Annotated[bool, typer.Option(help="Log the run to MLflow.")] = False,
    run_name: Annotated[str | None, typer.Option(help="MLflow run name.")] = None,
) -> None:
    """Run the golden set through one retrieval config: hit@k, recall, MRR, nDCG per slice."""
    raise typer.Exit(asyncio.run(_retrieval(golden, experiment, k, gate, mlflow, run_name)))


@app.command()
def ablation(
    golden: GoldenOpt = None,
    only: Annotated[
        list[str] | None, typer.Option("--only", help="Run only these configs (repeatable).")
    ] = None,
    k: Annotated[int, typer.Option(help="Cut-off for metrics.")] = 10,
    mlflow: Annotated[bool, typer.Option(help="Log every config as an MLflow run.")] = False,
) -> None:
    """Run every config of the experiments file and compare them to the first one (baseline)."""
    asyncio.run(_ablation(golden, only, k, mlflow))


def _experiments(settings: Settings) -> dict[str, RetrievalConfig]:
    path = Path(settings.eval.experiments_path)
    configs = load_experiments(path) if path.exists() else [RetrievalConfig()]
    return {c.name: c for c in configs}


async def _run_configs(
    settings: Settings, gs: GoldenSet, configs: list[RetrievalConfig], k: int
) -> tuple[IndexRecord, list[tuple[RetrievalConfig, RetrievalReport]]]:
    engine = make_engine(settings)
    qdrant = make_qdrant(settings)
    embedder, tei, cache = make_embedder(settings)
    results: list[tuple[RetrievalConfig, RetrievalReport]] = []
    try:
        index = await ensure_same_model(SqlIndexRegistry(engine), embedder.model_id)
        async with remote_rerankers(any(c.rerank for c in configs)):
            for cfg in configs:
                reranker = make_reranker(settings, cfg.rerank) if cfg.rerank else None
                retriever = Retriever(qdrant, embedder, cfg, reranker=reranker)
                console.print(f"[dim]running '{cfg.name}'...[/]")
                results.append((cfg, await evaluate_retrieval(gs.items, retriever, k=k)))
    finally:
        await tei.aclose()
        cache.close()
        await qdrant.close()
        await engine.dispose()
    return index, results


def _params(cfg: RetrievalConfig, index: IndexRecord, gs: GoldenSet, k: int) -> dict[str, Any]:
    return {
        **cfg.model_dump(exclude={"description"}),
        "embedding_model": index.model_id,
        "index_collection": index.collection,
        "index_fingerprint": index.fingerprint,
        "golden_version": gs.version,
        "golden_items": len(gs.items),
        "k": k,
    }


async def _retrieval(
    golden: Path | None,
    experiment: str | None,
    k: int,
    gate: bool,
    use_mlflow: bool,
    run_name: str | None,
) -> int:
    settings = get_settings()
    configure_logging(settings)
    gs = load_golden(golden or Path(settings.eval.golden_path))
    configs = _experiments(settings)
    if experiment and experiment not in configs:
        raise typer.BadParameter(f"unknown experiment {experiment!r}; known: {sorted(configs)}")
    # Default: the configuration the API serves, so the gate checks what is deployed.
    chosen = configs[experiment] if experiment else settings.retrieval

    index, [(cfg, report)] = await _run_configs(settings, gs, [chosen], k)

    markdown = to_markdown(report, title=f"Retrieval: {cfg.name} ({index.model_id}), k={k}")
    out_dir = Path(settings.eval.reports_dir)
    _write_reports(out_dir, report, markdown)
    console.print(Markdown(markdown))

    if use_mlflow:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        params = _params(cfg, index, gs, k)
        _log_mlflow(settings.eval, params, report, out_dir, run_name or f"{cfg.name}-{stamp}")

    if gate:
        failures = check_gate(report, Path(settings.eval.thresholds_path))
        if failures:
            console.print("[red bold]Quality gate FAILED[/]")
            for f in failures:
                console.print(f"[red]  - {f}[/]")
            return 1
        console.print("[green bold]Quality gate passed[/]")
    return 0


async def _ablation(golden: Path | None, only: list[str] | None, k: int, use_mlflow: bool) -> None:
    settings = get_settings()
    configure_logging(settings)
    gs = load_golden(golden or Path(settings.eval.golden_path))
    configs = [c for c in _experiments(settings).values() if not only or c.name in only]

    index, results = await _run_configs(settings, gs, configs, k)

    markdown = ablation_markdown(results, k)
    out_dir = Path(settings.eval.reports_dir)
    _write_ablation(out_dir, markdown)
    console.print(Markdown(markdown))
    if use_mlflow:
        for cfg, report in results:
            _write_reports(out_dir, report, to_markdown(report, title=cfg.name))
            _log_mlflow(settings.eval, _params(cfg, index, gs, k), report, out_dir, cfg.name)


def _write_ablation(out_dir: Path, markdown: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation_latest.md").write_text(markdown, encoding="utf-8")


def _write_reports(out_dir: Path, report: RetrievalReport, markdown: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_latest.json").write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "retrieval_latest.md").write_text(markdown, encoding="utf-8")


def _log_mlflow(
    cfg: Any, params: dict[str, Any], report: RetrievalReport, out_dir: Path, run_name: str
) -> None:
    import mlflow  # optional dependency group `eval`

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics({m.replace("@", "_at_"): v for m, v in report.overall.items()})
        for slice_kind, slices in (("cat", report.by_category), ("lang", report.by_lang)):
            for name, metrics in slices.items():
                mlflow.log_metrics(
                    {f"{slice_kind}.{name}.{m.replace('@', '_at_')}": v for m, v in metrics.items()}
                )
        mlflow.log_metrics({f"score.{k}": v for k, v in report.scores.items()})
        mlflow.log_metrics({f"latency_ms.{k}": v for k, v in report.latency_ms.items()})
        mlflow.log_artifact(str(out_dir / "retrieval_latest.json"))
        mlflow.log_artifact(str(out_dir / "retrieval_latest.md"))
    console.print(f"[dim]Logged to MLflow ({cfg.mlflow_tracking_uri}), run '{run_name}'.[/]")


@app.command()
def answers(
    golden: GoldenOpt = None,
    category: Annotated[
        list[str] | None, typer.Option("--category", "-c", help="Only these categories.")
    ] = None,
    item_id: Annotated[list[str] | None, typer.Option("--id", help="Only these items.")] = None,
    limit: Annotated[int | None, typer.Option(help="First N items (smoke runs).")] = None,
    model: Annotated[str | None, typer.Option(help="Generator override (LiteLLM id).")] = None,
    judge: Annotated[bool, typer.Option(help="Grade answers with the judge model.")] = True,
    cache: Annotated[bool, typer.Option(help="Reuse identical LLM calls (free reruns).")] = True,
    gate: Annotated[bool, typer.Option(help="Fail if below the `answers:` thresholds.")] = False,
    mlflow: Annotated[bool, typer.Option(help="Log the run to MLflow.")] = False,
    run_name: Annotated[str | None, typer.Option(help="MLflow run name.")] = None,
) -> None:
    """End to end: answers, refusals, citations, judge verdicts, latency and cost."""
    raise typer.Exit(
        asyncio.run(
            _answers(golden, category, item_id, limit, model, judge, cache, gate, mlflow, run_name)
        )
    )


async def _answers(
    golden: Path | None,
    categories: list[str] | None,
    ids: list[str] | None,
    limit: int | None,
    model: str | None,
    use_judge: bool,
    cache: bool,
    gate: bool,
    use_mlflow: bool,
    run_name: str | None,
) -> int:
    settings = get_settings()
    configure_logging(settings)
    gs = load_golden(golden or Path(settings.eval.golden_path))
    items = [
        i for i in gs.items
        if (not categories or i.category in categories) and (not ids or i.id in ids)
    ][:limit]  # fmt: skip

    done = 0

    def progress(r: AnswerItem) -> None:
        nonlocal done
        done += 1
        mark = "refused" if r.refused else (r.correctness or "answered")
        console.print(f"[dim]{done}/{len(items)} {r.id}: {mark}[/]")

    judge_llm, judge_cache = (
        make_llm(
            settings,
            model=settings.generation.judge_model,
            cache=cache,
            reasoning_effort=settings.generation.judge_reasoning_effort,
        )
        if use_judge
        else (None, None)
    )
    try:
        async with answerer_session(settings, model=model, cache=cache) as (answerer, _):
            report = await evaluate_answers(
                items,
                answerer,
                Judge(judge_llm) if judge_llm else None,
                concurrency=settings.generation.concurrency,
                on_item=progress,
            )
    finally:
        if judge_cache:
            judge_cache.close()

    markdown = answers_markdown(report, title=f"Answers: {report.model}")
    out_dir = Path(settings.eval.reports_dir)
    _write_answer_reports(out_dir, report, markdown)
    console.print(Markdown(markdown))

    if use_mlflow:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        params = {
            "model": report.model,
            "judge_model": report.judge_model,
            "judge_reasoning_effort": settings.generation.judge_reasoning_effort,
            "prompt_version": PROMPT_VERSION,
            "judge_version": JUDGE_VERSION,
            "retrieval": settings.retrieval.name,
            **settings.generation.model_dump(
                include={
                    "k",
                    "expand_chars",
                    "temperature",
                    "reasoning_effort",
                    "max_output_tokens",
                }
            ),
            "golden_version": gs.version,
            "golden_items": len(items),
        }
        _log_answers_mlflow(settings, params, report, out_dir, run_name or f"answers-{stamp}")

    if gate:
        failures = check_answers_gate(report, Path(settings.eval.thresholds_path))
        if failures:
            console.print("[red bold]Answer quality gate FAILED[/]")
            for f in failures:
                console.print(f"[red]  - {f}[/]")
            return 1
        console.print("[green bold]Answer quality gate passed[/]")
    return 0


def _write_answer_reports(out_dir: Path, report: AnswerReport, markdown: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "answers_latest.json").write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "answers_latest.md").write_text(markdown, encoding="utf-8")


def _log_answers_mlflow(
    settings: Settings, params: dict[str, Any], report: AnswerReport, out_dir: Path, run_name: str
) -> None:
    import mlflow

    mlflow.set_tracking_uri(settings.eval.mlflow_tracking_uri)
    mlflow.set_experiment(settings.eval.mlflow_experiment_answers)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(report.overall)
        for slice_kind, slices in (("cat", report.by_category), ("lang", report.by_lang)):
            for name, metrics in slices.items():
                mlflow.log_metrics({f"{slice_kind}.{name}.{m}": v for m, v in metrics.items()})
        mlflow.log_metrics({f"latency_ms.{k}": v for k, v in report.latency_ms.items()})
        mlflow.log_metrics({f"cost.{k}": v for k, v in report.cost.items()})
        mlflow.log_artifact(str(out_dir / "answers_latest.json"))
        mlflow.log_artifact(str(out_dir / "answers_latest.md"))
    console.print(f"[dim]Logged to MLflow, run '{run_name}'.[/]")


@app.command()
def cache(
    golden: GoldenOpt = None,
    pairs: Annotated[Path, typer.Option(help="Paraphrase / near-miss pairs.")] = Path(
        "eval/golden/cache_pairs_v1.yaml"
    ),
) -> None:
    """Choose the semantic-cache similarity threshold: paraphrase hits vs wrong-question hits."""
    asyncio.run(_cache(golden, pairs))


async def _cache(golden: Path | None, pairs_path: Path) -> None:
    settings = get_settings()
    configure_logging(settings)
    gs = load_golden(golden or Path(settings.eval.golden_path))
    pairs = load_pairs(pairs_path)
    for mismatch in check_lang_consistency(pairs):
        console.print(f"[yellow]language differs from its base: {mismatch}[/]")
    embedder, tei, emb_cache = make_embedder(settings)
    try:
        report = await evaluate_cache(pairs, gs.items, embedder)
    finally:
        await tei.aclose()
        emb_cache.close()
    markdown = cache_markdown(report)
    _write_cache_report(Path(settings.eval.reports_dir), report.as_dict(), markdown)
    console.print(Markdown(markdown))


def _write_cache_report(out_dir: Path, data: dict[str, Any], markdown: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cache_latest.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "cache_latest.md").write_text(markdown, encoding="utf-8")


@app.command()
def review(
    golden: GoldenOpt = None,
    reviewer: Annotated[str, typer.Option(help="Your name, stored with each decision.")] = "me",
    all_items: Annotated[
        bool, typer.Option("--all", help="Include already reviewed items.")
    ] = False,
) -> None:
    """Walk through draft items, show the source text of expected provisions, record a verdict."""
    asyncio.run(_review(golden, reviewer, all_items))


async def _review(golden: Path | None, reviewer: str, all_items: bool) -> None:
    settings = get_settings()
    path = golden or Path(settings.eval.golden_path)
    gs = load_golden(path)
    reviews = load_reviews(path)
    todo = [i for i in gs.items if all_items or status_of(i, reviews) == "draft"]
    console.print(f"{len(todo)} item(s) to review. Answers: [v]erify, [r]eject, [s]kip, [q]uit.\n")

    engine = make_engine(settings)
    try:
        for n, item in enumerate(todo, start=1):
            console.rule(f"[bold]{item.id}[/] ({n}/{len(todo)}) {item.category}, {item.lang}")
            console.print(f"[bold cyan]Q:[/] {item.question}")
            console.print(f"[bold green]Reference:[/] {item.reference}")
            async with engine.connect() as conn:
                for key in item.expected:
                    rows = (
                        await conn.execute(
                            select(ChunkRow.citation, ChunkRow.text, ChunkRow.provision_key)
                            .where(ChunkRow.lang == item.lang)
                            .where(
                                (ChunkRow.provision_key == key)
                                | ChunkRow.provision_key.startswith(f"{key}:")
                            )
                            .order_by(ChunkRow.ordinal)
                            .limit(3)
                        )
                    ).all()
                    for citation, text, _ in rows:
                        console.print(f"\n[yellow]{citation}[/]", markup=True)
                        console.print(text[:1500], markup=False, highlight=False)
            if not item.expected:
                console.print("[dim](no expected provision: the system should refuse)[/]")

            choice = typer.prompt("\nverdict [v/r/s/q]", default="s").strip().lower()
            if choice == "q":
                break
            if choice in {"v", "r"}:
                comment = typer.prompt("comment (optional)", default="", show_default=False)
                reviews[item.id] = Review(
                    status="verified" if choice == "v" else "rejected",
                    reviewer=reviewer,
                    date=date.today(),
                    comment=comment or None,
                )
                save_reviews(path, reviews)
    finally:
        await engine.dispose()
    done = sum(1 for i in gs.items if status_of(i, reviews) == "verified")
    console.print(f"\n{done}/{len(gs.items)} items verified. Decisions saved to reviews.yaml.")
