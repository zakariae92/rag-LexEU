"""A minimal embedding server with TEI's API, for machines TEI has no image for (ARM64).

    MODEL_DIR=/model uvicorn embed_server:from_env --factory --port 8081

HuggingFace text-embeddings-inference ships CPU images for x86 only. On CPU, TEI runs the model's
ONNX export with onnxruntime, then pools and normalises. This server does the same with the same
files (BGE-M3: CLS pooling, L2 normalisation), so its vectors match the index built with TEI and
nothing is re-embedded. It serves the subset of TEI's API the application uses: `POST /embed` and
`GET /health`.

Deployed on its own (docker/embeddings.Dockerfile): it must import nothing from `lexeu`.
"""

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

Vectors = npt.NDArray[np.float32]


class Encoder(Protocol):
    def encode(self, texts: list[str], truncate: bool) -> Vectors: ...


@dataclass(frozen=True)
class Limits:
    max_batch: int = 32  # inputs per request (TEI: --max-client-batch-size)
    max_tokens: int = 8192  # per input; longer ones are truncated or refused (TEI: auto-truncate)


class InputTooLong(ValueError):
    pass


class OnnxEncoder:
    """Tokenise, run the ONNX graph, pool on [CLS] (as the model's pooling config says)."""

    def __init__(self, model_dir: Path, max_tokens: int, threads: int | None = None) -> None:
        import onnxruntime as ort  # type: ignore[import-untyped]
        from tokenizers import Tokenizer

        pooling = json.loads((model_dir / "1_Pooling" / "config.json").read_text())
        if not pooling.get("pooling_mode_cls_token"):
            raise ValueError(f"only CLS pooling is implemented, got {pooling}")
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.no_truncation()
        self._pad_id = self._tokenizer.token_to_id("<pad>") or 0
        self._max_tokens = max_tokens

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or os.cpu_count() or 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(model_dir / "onnx" / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )

    def encode(self, texts: list[str], truncate: bool) -> Vectors:
        encodings = self._tokenizer.encode_batch(texts)
        ids = [e.ids for e in encodings]
        if any(len(i) > self._max_tokens for i in ids):
            if not truncate:
                raise InputTooLong(f"an input is longer than {self._max_tokens} tokens")
            # Keep the final </s>, as the tokenizer's own truncation would.
            ids = [
                i if len(i) <= self._max_tokens else i[: self._max_tokens - 1] + i[-1:] for i in ids
            ]
        width = max(len(i) for i in ids)
        input_ids = np.full((len(ids), width), self._pad_id, dtype=np.int64)
        mask = np.zeros((len(ids), width), dtype=np.int64)
        for row, seq in enumerate(ids):
            input_ids[row, : len(seq)] = seq
            mask[row, : len(seq)] = 1
        (tokens,) = self._session.run(
            ["token_embeddings"], {"input_ids": input_ids, "attention_mask": mask}
        )
        return np.asarray(tokens[:, 0, :], dtype=np.float32)  # [CLS]


def create_app(encoder: Encoder, limits: Limits | None = None) -> Starlette:
    limits = limits or Limits()
    # One inference at a time: onnxruntime already spreads one batch over every core, and
    # concurrent runs would only contend for them. Requests queue here instead.
    lock = asyncio.Lock()

    async def health(_: Request) -> Response:
        return Response(status_code=200)

    async def embed(request: Request) -> Response:
        try:
            body: Any = await request.json()
        except ValueError:
            return _error("invalid JSON", 422)
        inputs = body.get("inputs") if isinstance(body, dict) else None
        texts = [inputs] if isinstance(inputs, str) else inputs
        if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
            return _error("`inputs` must be a string or a non-empty list of strings", 422)
        if len(texts) > limits.max_batch:
            return _error(f"batch size {len(texts)} > maximum allowed {limits.max_batch}", 413)
        truncate = bool(body.get("truncate", True))
        async with lock:
            try:
                vectors = await run_in_threadpool(encoder.encode, texts, truncate)
            except InputTooLong as exc:
                return _error(str(exc), 413)
        if body.get("normalize", True):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)
        return JSONResponse(vectors.tolist())

    return Starlette(routes=[Route("/health", health), Route("/embed", embed, methods=["POST"])])


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)  # TEI's error shape


def from_env() -> Starlette:
    """The app configured from the environment (loads the model)."""
    limits = Limits(
        max_batch=int(os.environ.get("MAX_CLIENT_BATCH_SIZE", "32")),
        max_tokens=int(os.environ.get("MAX_INPUT_LENGTH", "8192")),
    )
    threads = int(os.environ["THREADS"]) if "THREADS" in os.environ else None
    encoder = OnnxEncoder(Path(os.environ.get("MODEL_DIR", "/model")), limits.max_tokens, threads)
    return create_app(encoder, limits)
