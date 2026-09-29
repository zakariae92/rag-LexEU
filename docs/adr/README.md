# Architecture Decision Records

Short records of significant technical decisions: the context, the choice, and what we gave up.
Format: [Michael Nygard's ADR template](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions).

| # | Decision | Status |
|---|---|---|
| [0001](0001-llm-serving.md) | Hybrid LLM serving: self-hosted open weights on serverless GPU + Gemini API behind LiteLLM | Accepted |
| [0002](0002-vector-store.md) | Qdrant as the vector store | Accepted |
| [0003](0003-object-storage.md) | RustFS as the local S3-compatible object store | Accepted |
| [0004](0004-ingestion-and-chunking.md) | Ingest from the Cellar API, chunk along the legal structure | Accepted |
| [0005](0005-embeddings-batch-and-online.md) | Embeddings: GPU batch path, CPU online path, shared cache, parity check | Accepted |
| [0006](0006-retrieval-pipeline.md) | Retrieval chosen by ablation: dense + recital demotion, no online reranker yet | Accepted |
| [0007](0007-generation.md) | Generation: cited sources, grounding checks, explicit refusals, flash-lite by measurement | Accepted |
