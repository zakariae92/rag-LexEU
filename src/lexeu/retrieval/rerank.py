"""Cross-encoder reranking of the retrieved candidates.

Evaluation runs the cross-encoders on a Modal GPU (`jobs/modal_rerank.py`). How to serve the
chosen one online (GPU endpoint vs a small model on CPU) depends on the latency measured in M3.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from lexeu.core.config import Settings
from lexeu.retrieval.search import Reranker


class ModalReranker:
    def __init__(self, model_id: str) -> None:
        from lexeu.jobs.modal_rerank import REVISIONS, GpuReranker

        if model_id not in REVISIONS:
            raise ValueError(f"unknown reranker {model_id!r}; pinned: {sorted(REVISIONS)}")
        self._model_id = model_id
        self._remote = GpuReranker(model_id=model_id)  # type: ignore[call-arg]

    @property
    def model_id(self) -> str:
        return self._model_id

    async def score(self, query: str, passages: list[str]) -> list[float]:
        return list(await self._remote.score.remote.aio(query, passages))


def make_reranker(settings: Settings, model_id: str) -> Reranker:
    return ModalReranker(model_id)


@asynccontextmanager
async def remote_rerankers(needed: bool) -> AsyncIterator[None]:
    """Start the Modal app only when a config actually reranks."""
    if not needed:
        yield
        return
    import modal

    from lexeu.jobs.modal_rerank import app

    with modal.enable_output():
        async with app.run():
            yield
