"""`lexeu keys create | list | revoke`: manage API keys."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from lexeu.core.config import get_settings
from lexeu.infra.api_keys import ApiKeyStore
from lexeu.infra.resources import make_engine

app = typer.Typer(help="Manage API keys.", no_args_is_help=True)
console = Console()


def _run[T](fn: Callable[[ApiKeyStore], Awaitable[T]]) -> T:
    async def main() -> T:
        engine = make_engine(get_settings())
        try:
            return await fn(ApiKeyStore(engine))
        finally:
            await engine.dispose()

    return asyncio.run(main())


@app.command()
def create(
    name: Annotated[str, typer.Argument(help="Who the key is for, e.g. 'web-ui' or 'alice'.")],
    rate_limit: Annotated[
        int | None, typer.Option(help="Requests per minute (default from settings).")
    ] = None,
) -> None:
    """Create a key. The secret is printed once: store it, it cannot be shown again."""
    limit = rate_limit or get_settings().auth.default_rate_limit_per_min
    secret, key = _run(lambda store: store.create(name, limit))
    console.print(f"Created key [bold]{key.name}[/] ({key.prefix}..., {limit} requests/min):")
    console.print(secret, soft_wrap=True, highlight=False)
    console.print("[yellow]Store it now: only its hash is kept.[/]")


@app.command("list")
def list_keys() -> None:
    """List keys (prefixes only)."""
    keys = _run(lambda store: store.list())
    table = Table("name", "prefix", "requests/min", "created", "status")
    for k in keys:
        created = k.created_at.strftime("%Y-%m-%d") if k.created_at else ""
        status = f"revoked {k.revoked_at:%Y-%m-%d}" if k.revoked_at else "active"
        table.add_row(k.name, f"{k.prefix}...", str(k.rate_limit_per_min), created, status)
    console.print(table)


@app.command()
def revoke(name: Annotated[str, typer.Argument(help="Name of the key to revoke.")]) -> None:
    """Revoke a key: it stops working on the next request."""
    if not _run(lambda store: store.revoke(name)):
        console.print(f"[red]No active key named {name!r}.[/]")
        raise typer.Exit(1)
    console.print(f"Revoked [bold]{name}[/].")
