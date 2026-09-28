# 0002: Qdrant as the vector store

**Status:** Accepted

## Context

Legal retrieval needs both semantic matching ("lawful basis") and exact matching ("Article 6(1)(f)",
"DPIA"). Queries must also filter by regulation, language and version, and re-indexing
(new chunker or new embedding model) must not cause downtime.

## Decision

Use **Qdrant**.

- Native **hybrid search**: dense + sparse vectors in one query with server-side fusion. This pairs
  with BGE-M3, which emits both kinds of vector.
- **Payload filtering and indexes** on `celex`, `lang`, `article`, `version_date`.
- **Collection aliases**: build `chunks_vN+1` next to the live index, evaluate it, then switch the
  alias atomically. Rollback means switching the alias back.
- Single Rust binary, small footprint, and a free managed tier for deployment.

## Consequences

- One more service to run (versus pgvector inside Postgres).
- Rejected: **pgvector** (less mature hybrid search and sparse vectors), **Pinecone** (proprietary,
  paid), **Weaviate** (heavier to run for the same features).
