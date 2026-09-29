"""`lexeu ask "question"`: one grounded answer from the command line."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import typer
from qdrant_client import AsyncQdrantClient
from rich.console import Console
from rich.markdown import Markdown
from sqlalchemy.ext.asyncio import create_async_engine

from lexeu.commands.index import make_embedder
from lexeu.core.config import Settings, get_settings
from lexeu.core.logging import configure_logging
from lexeu.generation.answer import Answerer
from lexeu.generation.factory import MissingApiKeyError, make_llm
from lexeu.generation.llm import LlmClient
from lexeu.infra.db import SqlIndexRegistry
from lexeu.retrieval.index import ensure_same_model
from lexeu.retrieval.search import Retriever
from lexeu.retrieval.sparse import detect_lang

console = Console()


@asynccontextmanager
async def answerer_session(
    settings: Settings, model: str | None = None, cache: bool = False
) -> AsyncIterator[tuple[Answerer, LlmClient]]:
    """Answerer wired to the live index, closing every client on exit."""
    try:
        llm, llm_cache = make_llm(settings, model=model, cache=cache)
    except MissingApiKeyError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(2) from exc
    engine = create_async_engine(settings.postgres.dsn)
    qdrant = AsyncQdrantClient(url=settings.qdrant.url)
    embedder, tei, emb_cache = make_embedder(settings)
    try:
        await ensure_same_model(SqlIndexRegistry(engine), embedder.model_id)
        retriever = Retriever(qdrant, embedder, settings.retrieval)
        detect_lang("warm-up")  # load the language models once, outside measured latencies
        yield (
            Answerer(
                retriever,
                llm,
                k=settings.generation.k,
                expand_chars=settings.generation.expand_chars,
            ),
            llm,
        )
    finally:
        await tei.aclose()
        emb_cache.close()
        if llm_cache:
            llm_cache.close()
        await qdrant.close()
        await engine.dispose()


def ask(
    question: Annotated[str, typer.Argument(help="Your question, in English or French.")],
    lang: Annotated[str | None, typer.Option(help="Force the answer language (en|fr).")] = None,
    model: Annotated[str | None, typer.Option(help="LiteLLM model id override.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the raw answer object.")] = False,
) -> None:
    """Answer a question from the corpus, with citations (or refuse)."""
    asyncio.run(_ask(question, lang, model, as_json))


async def _ask(question: str, lang: str | None, model: str | None, as_json: bool) -> None:
    settings = get_settings()
    configure_logging(settings)
    async with answerer_session(settings, model=model) as (answerer, _):
        answer = await answerer.answer(question, lang=lang)
    if as_json:
        console.print_json(json.dumps(answer.as_dict(), ensure_ascii=False, default=str))
        return
    console.print(Markdown(answer.text))
    if answer.citations:
        console.print()
        for c in answer.citations:
            console.print(f"[dim][{c.n}] {c.citation}  ({c.provision_key}, {c.lang})[/]")
    t = answer.timings_ms
    console.print(
        f"\n[dim]{answer.model} | retrieval {t.get('retrieval', 0):.0f} ms, generation "
        f"{t.get('generation', 0):.0f} ms | {answer.input_tokens}+{answer.output_tokens} tokens, "
        f"${answer.cost_usd:.5f}"
        + (f" | refused: {answer.refusal_reason}" if answer.refused else "")
        + "[/]"
    )
