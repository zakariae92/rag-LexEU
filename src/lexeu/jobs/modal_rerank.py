"""Cross-encoder reranking on a Modal GPU (used to evaluate rerankers; serving is decided later).

Scores go through a sigmoid so they land in [0, 1]: comparable across queries, which is what a
refusal threshold needs (M4). Model revisions are pinned, like the embedding model.
"""

import modal

REVISIONS = {
    "BAAI/bge-reranker-v2-m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1": "1427fd652930e4ba29e8149678df786c240d8825",
}
# Longer inputs than the model's position embeddings crash the CUDA kernel (device-side assert).
MAX_LENGTH = {
    "BAAI/bge-reranker-v2-m3": 1024,
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1": 512,
}

app = modal.App("rag-lexeu-rerank")
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
    scaledown_window=120,
    max_containers=1,
)
class GpuReranker:
    model_id: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            self.model_id,
            revision=REVISIONS[self.model_id],
            device="cuda",
            max_length=MAX_LENGTH[self.model_id],
            activation_fn=torch.nn.Sigmoid(),
        )
        hf_cache.commit()

    @modal.method()
    def score(self, query: str, passages: list[str]) -> list[float]:
        scores = self.model.predict([(query, p) for p in passages], batch_size=32)
        return [float(s) for s in scores]
