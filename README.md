# rag-LexEU

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**A production-grade, bilingual (EN/FR) RAG assistant over EU regulations**: GDPR, AI Act, DORA,
NIS2, DSA and Data Act. Every answer cites the exact article, and the assistant says *"I don't know"*
when the law doesn't cover the question.

The project is built around the questions a team asks before shipping a RAG system:

| Question | How rag-LexEU answers it |
|---|---|
| Did you build an eval set first? | Hand-verified golden set, evaluated in CI on every PR |
| How good is retrieval? | Hybrid search (dense + sparse) + cross-encoder reranking, with Recall@k/MRR tracked in MLflow |
| What stops hallucinations? | Answers use retrieved context only, with article-level citations, a grounding check and refusal under a confidence threshold |
| Same embedding model on both sides? | Model id is stored with every vector, checked at startup, and re-indexing uses blue/green aliases |
| Latency? | p95 measured per stage with k6 load tests |
| Cost per query? | Semantic cache, query routing to cheaper models, LiteLLM cost tracking |
| How do you know prod broke? | Every query traced (Langfuse + OpenTelemetry), with alerts on faithfulness drops and cost spikes |

## Architecture

```
 EUR-Lex (EN+FR) ─┐
 EDPB guidelines ─┼─► Prefect ingestion ─► object store (raw) ─► legal-aware chunker ─► BGE-M3 ─► Qdrant
                  ┘                                                                              │
 User ─► Next.js ─► FastAPI ─► guardrails ─► semantic cache (Redis) ─► hybrid search + rerank ◄──┘
                                   │                                          │
                                   └──────────► LiteLLM ─► vLLM on Modal (Qwen3) / Gemini
                                                   │
                      Langfuse · Prometheus · Grafana · MLflow · Postgres (feedback, doc registry)
```

Key decisions and their trade-offs are recorded in [`docs/adr/`](docs/adr/).

## Roadmap

- [x] **M0 · Foundations**: uv project, typed settings, structured logging, FastAPI with health/readiness probes, Docker Compose stack, CI (lint, types, tests, container smoke test)
- [ ] **M1 · Ingestion**: EUR-Lex connector (EN+FR), legal-aware chunking, document registry, incremental sync
- [ ] **M2 · Evaluation first**: golden set, eval harness, MLflow, baseline scores
- [ ] **M3 · Retrieval**: hybrid search, reranking, cross-reference expansion, ablation table
- [ ] **M4 · Generation**: LiteLLM gateway, citations, refusal, grounding check
- [ ] **M5 · Backend**: streaming chat API, auth, rate limiting, semantic cache, feedback
- [ ] **M6 · Observability**: Langfuse, OpenTelemetry, Grafana dashboards, alerts, load tests
- [ ] **M7 · UI & delivery**: Next.js chat, CI eval gate, Terraform deployment
- [ ] **M8 · Results**: ablation tables, demo video

## Quickstart

Requirements: [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync                          # install Python 3.12 + dependencies from uv.lock
cp .env.example .env             # then edit the secrets
uv run poe up                    # start Qdrant, Postgres, Redis, object store
uv run poe dev                   # API with hot reload on http://localhost:8000
```

Then check <http://localhost:8000/ready> (every dependency should be `up`) and the interactive docs at
<http://localhost:8000/docs>.

| Task | Command |
|---|---|
| Lint + format check + strict type-check | `uv run poe lint` |
| Tests with coverage | `uv run poe test` |
| Full stack in containers (incl. API) | `uv run poe up-all` |
| Stop everything | `uv run poe down` |
| Install git hooks | `uv run pre-commit install` |

## Project layout

```
src/lexeu/
  api/          FastAPI app factory, middleware, routes
  core/         settings, logging
  infra/        clients for Postgres, Qdrant, Redis, object store
tests/          unit tests (integration tests from M1)
docker/         container images
docs/adr/       architecture decision records
```

## License

MIT
