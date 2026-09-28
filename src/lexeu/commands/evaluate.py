"""`lexeu eval validate | retrieval | review`."""

import asyncio
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from qdrant_client import AsyncQdrantClient
from rich.console import Console
from rich.markdown import Markdown
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from lexeu.commands.index import make_embedder
from lexeu.core.config import get_settings
from lexeu.core.logging import configure_logging
from lexeu.eval.golden import (
    Review,
    load_golden,
    load_reviews,
    save_reviews,
    status_of,
)
from lexeu.eval.metrics import matches
from lexeu.eval.retrieval import RetrievalReport, check_gate, evaluate_retrieval, to_markdown
from lexeu.infra.db import ChunkRow, SqlIndexRegistry
from lexeu.retrieval.index import ensure_same_model
from lexeu.retrieval.search import DenseRetriever

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
    engine = create_async_engine(settings.postgres.dsn)
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


@app.command()
def retrieval(
    golden: GoldenOpt = None,
    k: Annotated[int, typer.Option(help="Cut-off for metrics.")] = 10,
    gate: Annotated[bool, typer.Option(help="Fail if below eval/thresholds.yaml.")] = False,
    mlflow: Annotated[bool, typer.Option(help="Log the run to MLflow.")] = False,
    run_name: Annotated[str | None, typer.Option(help="MLflow run name.")] = None,
) -> None:
    """Run the golden set through the retriever: hit@k, recall, MRR, nDCG per slice."""
    raise typer.Exit(asyncio.run(_retrieval(golden, k, gate, mlflow, run_name)))


async def _retrieval(
    golden: Path | None, k: int, gate: bool, use_mlflow: bool, run_name: str | None
) -> int:
    settings = get_settings()
    configure_logging(settings)
    path = golden or Path(settings.eval.golden_path)
    gs = load_golden(path)

    engine = create_async_engine(settings.postgres.dsn)
    qdrant = AsyncQdrantClient(url=settings.qdrant.url)
    embedder, tei, cache = make_embedder(settings)
    try:
        index = await ensure_same_model(SqlIndexRegistry(engine), embedder.model_id)
        report = await evaluate_retrieval(gs.items, DenseRetriever(qdrant, embedder), k=k)
    finally:
        await tei.aclose()
        cache.close()
        await qdrant.close()
        await engine.dispose()

    markdown = to_markdown(report, title=f"Retrieval: dense {index.model_id}, k={k}")
    out_dir = Path(settings.eval.reports_dir)
    _write_reports(out_dir, report, markdown)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    console.print(Markdown(markdown))

    params = {
        "retriever": "dense",
        "embedding_model": index.model_id,
        "index_collection": index.collection,
        "index_fingerprint": index.fingerprint,
        "golden_version": gs.version,
        "golden_items": len(gs.items),
        "k": k,
    }
    if use_mlflow:
        _log_mlflow(settings.eval, params, report, out_dir, run_name or f"dense-{stamp}")

    if gate:
        failures = check_gate(report, Path(settings.eval.thresholds_path))
        if failures:
            console.print("[red bold]Quality gate FAILED[/]")
            for f in failures:
                console.print(f"[red]  - {f}[/]")
            return 1
        console.print("[green bold]Quality gate passed[/]")
    return 0


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

    engine = create_async_engine(settings.postgres.dsn)
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
