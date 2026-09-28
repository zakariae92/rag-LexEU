"""`lexeu index embed | build | status`: embeddings and the vector index behind `chunks`."""

import asyncio
import random
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from qdrant_client import AsyncQdrantClient
from rich.console import Console
from sqlalchemy.ext.asyncio import create_async_engine

from lexeu.core.config import EmbeddingSettings, Settings, get_settings
from lexeu.core.logging import configure_logging
from lexeu.infra.db import SqlIndexRegistry
from lexeu.retrieval.embeddings import (
    CachedEmbedder,
    EmbeddingCache,
    TeiEmbedder,
    Vectors,
    min_cosine,
)
from lexeu.retrieval.index import build_index, embedding_text

app = typer.Typer(help="Build and inspect the vector index.", no_args_is_help=True)
console = Console()


def make_embedder(settings: Settings) -> tuple[CachedEmbedder, TeiEmbedder, EmbeddingCache]:
    cfg = settings.embeddings
    tei = TeiEmbedder(cfg.url, cfg.model_id, batch_size=cfg.batch_size)
    cache = EmbeddingCache(Path(cfg.cache_path))
    return CachedEmbedder(tei, cache), tei, cache


PARITY_MIN_COSINE = 0.999


@app.command()
def embed(
    batch: Annotated[int, typer.Option(help="Texts per remote call.")] = 256,
    parity_sample: Annotated[int, typer.Option(help="Texts compared GPU vs TEI first.")] = 8,
    skip_parity: Annotated[
        bool, typer.Option(help="Do not compare with TEI (not advised).")
    ] = False,
) -> None:
    """Embed every chunk missing from the cache on a Modal GPU (batch path)."""
    asyncio.run(_embed(batch, parity_sample, skip_parity))


async def _embed(batch: int, parity_sample: int, skip_parity: bool) -> None:
    settings = get_settings()
    configure_logging(settings)
    cfg = settings.embeddings
    engine = create_async_engine(settings.postgres.dsn)
    try:
        chunks = await SqlIndexRegistry(engine).all_chunks()
    finally:
        await engine.dispose()

    cache = EmbeddingCache(Path(cfg.cache_path))
    texts = list(dict.fromkeys(embedding_text(c) for c in chunks))
    keys = {t: EmbeddingCache.key(cfg.model_id, t) for t in texts}
    cached = cache.get_many(list(keys.values()))
    todo = [t for t in texts if keys[t] not in cached]
    console.print(f"{len(texts)} texts, {len(texts) - len(todo)} cached, {len(todo)} to embed")
    if not todo:
        cache.close()
        return

    import truststore  # modal's gRPC client must trust the OS store (proxies, antivirus TLS)

    truststore.inject_into_ssl()
    import modal

    from lexeu.jobs.modal_embed import app as modal_app

    try:
        with modal.enable_output():
            async with modal_app.run():
                await _embed_remote(cfg, todo, keys, cache, batch, parity_sample, skip_parity)
        console.print(f"[green]Done.[/] Cache now holds {len(cache)} vectors ({cfg.cache_path}).")
    finally:
        cache.close()


async def _embed_remote(
    cfg: EmbeddingSettings,
    todo: list[str],
    keys: dict[str, str],
    cache: EmbeddingCache,
    batch: int,
    parity_sample: int,
    skip_parity: bool,
) -> None:
    from lexeu.jobs.modal_embed import GpuEmbedder

    # modal.parameter fields become constructor kwargs at runtime
    gpu = GpuEmbedder(model_id=cfg.model_id, revision=cfg.revision)  # type: ignore[call-arg]

    def to_vectors(raw: bytes, n: int) -> Vectors:
        return np.frombuffer(raw, dtype=np.float32).reshape(n, -1)

    if not skip_parity:
        rng = random.Random(0)  # noqa: S311 - reproducible sample, not security-related
        sample = rng.sample(todo, min(parity_sample, len(todo)))
        tei = TeiEmbedder(cfg.url, cfg.model_id, batch_size=cfg.batch_size)
        try:
            served = await tei.served_model()
            if served != cfg.model_id:
                raise typer.BadParameter(f"TEI serves {served!r}, expected {cfg.model_id!r}")
            gpu_vectors = to_vectors(await gpu.embed.remote.aio(sample), len(sample))
            parity = min_cosine(gpu_vectors, await tei.embed(sample))
        finally:
            await tei.aclose()
        console.print(f"parity GPU vs TEI on {len(sample)} texts: min cosine = {parity:.5f}")
        if parity < PARITY_MIN_COSINE:
            console.print(f"[red]Parity below {PARITY_MIN_COSINE}: not writing to the cache.[/]")
            raise typer.Exit(1)

    batches = [todo[i : i + batch] for i in range(0, len(todo), batch)]
    n = 0
    async for raw in gpu.embed.map.aio(batches):  # results come back in input order
        batch_texts = batches[n]
        vectors = to_vectors(raw, len(batch_texts))
        cache.put_many({keys[t]: v for t, v in zip(batch_texts, vectors, strict=True)})
        n += 1
        console.print(f"  batch {n}/{len(batches)} stored")


@app.command()
def build() -> None:
    """Embed every chunk (cached) and switch the `chunks` alias to the new collection."""
    asyncio.run(_build())


async def _build() -> None:
    settings = get_settings()
    configure_logging(settings)
    engine = create_async_engine(settings.postgres.dsn)
    qdrant = AsyncQdrantClient(url=settings.qdrant.url)
    embedder, tei, cache = make_embedder(settings)
    try:
        report = await build_index(SqlIndexRegistry(engine), qdrant, embedder)
    finally:
        await tei.aclose()
        cache.close()
        await qdrant.close()
        await engine.dispose()
    console.print(
        f"[bold]{report.action}[/] {report.collection}: {report.n_points} points, "
        f"embeddings {report.embed_hits} cached / {report.embed_misses} computed, "
        f"{report.duration_s}s"
    )


@app.command()
def status() -> None:
    """Which collection is live, and with which embedding model."""
    asyncio.run(_status())


async def _status() -> None:
    settings = get_settings()
    engine = create_async_engine(settings.postgres.dsn)
    try:
        active = await SqlIndexRegistry(engine).active("chunks")
    finally:
        await engine.dispose()
    if active is None:
        console.print("[yellow]No active index. Run `lexeu index build`.[/]")
        raise typer.Exit(1)
    cache = EmbeddingCache(Path(settings.embeddings.cache_path))
    console.print(
        f"alias [bold]chunks[/] -> {active.collection}\n"
        f"model {active.model_id} (dim {active.dim}), {active.n_points} points, "
        f"fingerprint {active.fingerprint}\n"
        f"embedding cache: {len(cache)} vectors in {settings.embeddings.cache_path}"
    )
    cache.close()
