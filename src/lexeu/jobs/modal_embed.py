"""Corpus embedding on a Modal GPU (batch path).

Online, queries are embedded by TEI on CPU (~100 ms each). Offline, embedding ~6k chunks on a
laptop CPU takes ~1h40, so the batch runs here on a GPU in a couple of minutes instead.

Both paths must produce the same vectors, otherwise query/document similarity silently drifts:
- same model and pinned revision,
- same pooling (CLS) and L2 normalisation as TEI for BGE-M3,
- fp32 on the GPU (no fp16) so the numbers match the ONNX fp32 model TEI runs.
`lexeu index embed` checks this parity on a sample before trusting the GPU results.
"""

import modal

MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

app = modal.App("rag-lexeu-embed")
hf_cache = modal.Volume.from_name("rag-lexeu-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", "sentence-transformers==6.1.0", "transformers==5.17.0")
    .env({"HF_HOME": "/cache", "HF_HUB_DISABLE_TELEMETRY": "1"})
)


@app.cls(
    gpu="L4",
    image=image,
    volumes={"/cache": hf_cache},
    timeout=1800,
    scaledown_window=60,
    max_containers=2,  # each container loads 2.2 GB of weights: parallelism has a cost
)
class GpuEmbedder:
    model_id: str = modal.parameter(default=MODEL_ID)
    revision: str = modal.parameter(default=MODEL_REVISION)

    @modal.enter()
    def load(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            self.model_id,
            revision=self.revision,
            device="cuda",
            model_kwargs={"torch_dtype": "float32"},
        )
        self.model.max_seq_length = 4096  # = TEI max_input_length (bounded by --max-batch-tokens)
        hf_cache.commit()

    @modal.method()
    def embed(self, texts: list[str]) -> bytes:
        """Returns float32 row-major bytes (n x dim): compact and exact over the wire."""
        vectors = self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True
        )
        return bytes(vectors.astype("float32").tobytes())
