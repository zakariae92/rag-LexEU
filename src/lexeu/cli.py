"""Command line: `lexeu ingest | corpus | index | eval | ask`."""

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy.ext.asyncio import create_async_engine

from lexeu.commands import ask as ask_cmd
from lexeu.commands import evaluate as evaluate_cmd
from lexeu.commands import index as index_cmd
from lexeu.core.config import get_settings
from lexeu.core.logging import configure_logging
from lexeu.infra.db import SqlRegistry
from lexeu.infra.storage import S3RawStore
from lexeu.ingestion.corpus import load_corpus
from lexeu.ingestion.fetcher import CellarFetcher
from lexeu.ingestion.pipeline import sync

app = typer.Typer(help="rag-LexEU command line.", no_args_is_help=True)


@app.callback()
def _setup() -> None:
    """Every HTTPS client in the CLI (Cellar, HuggingFace, Modal) trusts the OS certificate
    store: works behind corporate proxies and antivirus TLS inspection."""
    from lexeu.core.tls import use_system_trust

    use_system_trust()


corpus_app = typer.Typer(help="Inspect the ingested corpus.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")
app.add_typer(index_cmd.app, name="index")
app.add_typer(evaluate_cmd.app, name="eval")
app.command()(ask_cmd.ask)
console = Console()


@app.command()
def ingest(
    celex: Annotated[
        list[str] | None, typer.Option("--celex", "-c", help="Only these acts (repeatable).")
    ] = None,
    force: Annotated[bool, typer.Option(help="Re-process even if unchanged.")] = False,
) -> None:
    """Sync the corpus: fetch from Cellar, archive raw, parse, chunk, register."""
    raise typer.Exit(asyncio.run(_ingest(set(celex) if celex else None, force)))


async def _ingest(only: set[str] | None, force: bool) -> int:
    settings = get_settings()
    configure_logging(settings)
    engine = create_async_engine(settings.postgres.dsn)
    registry = SqlRegistry(engine)
    store = S3RawStore(settings.object_store)
    fetcher = CellarFetcher(settings.ingestion)
    try:
        await store.ensure_bucket()
        run_id = await registry.start_run()
        report = await sync(
            load_corpus(),
            fetcher,
            store,
            registry,
            max_chars=settings.ingestion.chunk_max_chars,
            max_concurrency=settings.ingestion.max_concurrency,
            only=only,
            force=force,
        )
        await registry.finish_run(run_id, "succeeded" if report.ok else "failed", report.as_dict())
    finally:
        await fetcher.aclose()
        await engine.dispose()

    table = Table(title=f"Ingestion run #{run_id}")
    for col in ("CELEX", "lang", "action", "chunks", "ms", "error"):
        table.add_column(col)
    colors = {"created": "green", "updated": "yellow", "failed": "red", "deleted": "magenta"}
    for r in sorted(report.results, key=lambda r: (r.celex, r.lang)):
        color = colors.get(r.action, "white")
        table.add_row(
            r.celex, r.lang, f"[{color}]{r.action}[/]", str(r.n_chunks),
            f"{r.duration_ms:.0f}", r.error or "",
        )  # fmt: skip
    console.print(table)
    console.print(report.counts)
    return 0 if report.ok else 1


@corpus_app.command("status")
def status() -> None:
    """What is in the registry, per document."""
    asyncio.run(_status())


async def _status() -> None:
    settings = get_settings()
    engine = create_async_engine(settings.postgres.dsn)
    try:
        docs = await SqlRegistry(engine).list_documents()
    finally:
        await engine.dispose()
    corpus = load_corpus()
    table = Table(title="Corpus registry")
    for col in ("CELEX", "act", "lang", "chunks", "sha256", "pipeline"):
        table.add_column(col)
    for d in docs:
        name = corpus.get(d.celex).short_name.get(d.lang, "?")  # type: ignore[call-overload]
        table.add_row(
            d.celex, name, d.lang, str(d.n_chunks), d.content_sha256[:12], d.pipeline_version
        )
    console.print(table)
    console.print(f"{len(docs)} documents, {sum(d.n_chunks for d in docs)} chunks")


@corpus_app.command("show")
def show(
    celex: Annotated[str, typer.Argument(help="e.g. 32016R0679")],
    eli_id: Annotated[str, typer.Argument(help="e.g. art_6, rct_12, anx_III")],
    lang: Annotated[str, typer.Option("--lang", "-l")] = "en",
) -> None:
    """Print the chunks of one provision, as the retriever will see them."""
    asyncio.run(_show(celex, eli_id, lang))


async def _show(celex: str, eli_id: str, lang: str) -> None:
    settings = get_settings()
    engine = create_async_engine(settings.postgres.dsn)
    try:
        chunks = await SqlRegistry(engine).get_chunks(celex, lang, eli_id)
    finally:
        await engine.dispose()
    if not chunks:
        console.print(f"[red]No chunks for {celex} {eli_id} ({lang})[/]")
        raise typer.Exit(1)
    for c in chunks:
        console.rule(f"[bold]{c.citation}[/]  [dim]{c.chunk_id}[/]")
        console.print(f"[cyan]{c.header}[/]\n")
        console.print(c.text, markup=False, highlight=False)
        console.print()


if __name__ == "__main__":
    app()
