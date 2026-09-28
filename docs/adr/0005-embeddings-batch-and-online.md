# 0005: Embeddings with a GPU batch path, a CPU online path, and a shared cache

**Status:** Accepted

## Context

BGE-M3 (568M parameters, 1024-d, multilingual) is the embedding model. Measured on the
development laptop (i5-1145G7, no CUDA GPU, 6 GB Docker VM):

| Workload | Throughput / latency |
|---|---|
| TEI CPU in Docker, corpus batch | 0.5 chunks/s (ONNX backend caps batches at 8; VM near its memory limit) |
| ONNX Runtime native, 8 threads | ~1 chunk/s, i.e. ~1h40 for 6,111 chunks |
| TEI CPU, one query | p50 ~100 ms |

Queries are fine on CPU. The corpus is not. The default TEI warm-up (16k-token batches) was also
OOM-killed at 6 GB.

## Decision

- **Online path:** HuggingFace **TEI** (CPU image, compose profile `ml`) embeds queries, with
  `--max-batch-tokens 4096` to fit the VM.
- **Batch path:** `lexeu index embed` sends only **cache misses** to a **Modal GPU** (L4) running
  sentence-transformers. The container scales to zero after the job; a full corpus run costs cents.
- **One cache** (SQLite, `sha256(model_id + text)` to float32) sits between the two. The index
  build only reads the cache, so it is instant and machine-independent.
- **Same vector space, enforced:**
  - model **revision pinned** on both paths (`5617a9f…`);
  - CLS pooling and L2 normalisation (checked with TEI `/info`);
  - **fp32 on the GPU**;
  - same truncation limit (4096 tokens);
  - a **parity check** before any batch is cached (min cosine between GPU and TEI vectors on a
    sample must be at least 0.999; otherwise nothing is written);
  - at query time, `ensure_same_model` refuses to search an index built with another model.

## Consequences

- Re-embedding after a parser or chunking change costs cents and minutes, not hours.
- The CI eval job restores the cache (GitHub Actions cache) and only calls Modal for misses. It
  needs `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` as repository secrets.
- Two runtimes (ONNX/TEI and PyTorch) execute the same model. The parity check is what makes this
  safe, and it runs on every batch job.
- Rejected: torch in the API image (~2 GB image, slow on CPU anyway); embedding on the laptop
  (hours); an always-on GPU (cost).
