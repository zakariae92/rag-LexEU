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
- [x] **M1 · Ingestion**: official Cellar API (EN+FR), structure-aware chunking with citations, Postgres registry with lineage, incremental sync, Alembic migrations
- [x] **M2 · Evaluation first**: 165-question golden set (EN/FR), retrieval metrics, MLflow, BGE-M3 embeddings (GPU batch + CPU online, parity-checked), blue/green Qdrant index, CI quality gate
- [x] **M3 · Retrieval**: 14-config ablation (dense, BM25 hybrid, recital policy, cross-encoder rerankers), production config chosen by the numbers, CI gate raised
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
uv run poe migrate               # create the registry tables
uv run poe ingest                # fetch + parse + chunk the 6 acts in EN and FR (~5 s)
```

Then check <http://localhost:8000/ready> (every dependency should be `up`) and the interactive docs at
<http://localhost:8000/docs>.

| Task | Command |
|---|---|
| Lint + format check + strict type-check | `uv run poe lint` |
| Unit tests with coverage | `uv run poe test` |
| Integration tests (real Postgres + S3) | `uv run poe test-it` |
| Smoke-test every running service | `uv run poe smoke` |
| Full stack in containers (incl. API) | `uv run poe up-all` |
| Stop everything | `uv run poe down` |
| Install git hooks | `uv run pre-commit install` |

## Ingestion (M1)

```
corpus.toml ─► Cellar API ─► raw/{celex}/{lang}/{sha256}.xhtml (S3) ─► parser ─► chunker ─► Postgres
                (EN + FR)        content-addressed, never overwritten    ELI structure   citations
```

```bash
uv run lexeu ingest                     # incremental: only new or changed documents are processed
uv run lexeu ingest -c 32016R0679 --force
uv run lexeu corpus status              # documents, chunk counts, hashes, pipeline version
uv run lexeu corpus show 32016R0679 art_6 --lang fr
```

| | |
|---|---|
| Corpus | GDPR, AI Act, DORA, NIS2, DSA, Data Act in English and French: 12 documents |
| Chunks | ~6,100: one per paragraph, recital or annex, split between points above 1,800 chars |
| Citation per chunk | `Art. 6(1) GDPR`, `Art. 6, par. 1, RGPD`, `Art. 4 GDPR, points (5)–(9)` |
| EN ↔ FR alignment | 100% of provisions share a language-neutral `provision_key` |
| Full sync / no-op re-sync | ~4 s / ~1 s |

Design and trade-offs: [ADR 0004](docs/adr/0004-ingestion-and-chunking.md).

## Evaluation (M2)

Measure first, tune second. The golden set (`eval/golden/golden_v1.yaml`) holds **165 questions**
(31 in French) with the provisions that answer them: lookups, definitions, obligations, dates,
cross-regulation comparisons, **12 questions the corpus cannot answer** and **7 false premises**
("What does Article 120 GDPR say?"). Items are drafts until reviewed with `lexeu eval review`.

**Baseline:** dense retrieval with BGE-M3, top-10 distinct provisions ([full report](eval/baselines/m2_dense_bge-m3.md)).

| | hit@1 | hit@5 | hit@10 | recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| overall (151) | 0.530 | 0.881 | 0.940 | 0.917 | 0.687 | 0.730 |
| English (123) | 0.553 | 0.870 | 0.943 | 0.919 | 0.700 | 0.739 |
| French (28) | 0.429 | 0.929 | 0.929 | 0.911 | 0.628 | 0.693 |

What the numbers say, and the hypotheses for M3:

- **Recitals outrank articles.** In 47 of the 71 top-1 misses, a recital (explanatory language,
  close to how people ask) beats the article that holds the rule. Test: rank articles first,
  or index recitals separately.
- **Definitions, dates and cross-regulation questions are the weak slices** (hit@10 0.80-0.85).
  Exact terms ("deployer", "17 February 2024") call for **hybrid search** (sparse + dense).
- **A raw similarity threshold cannot drive refusals.** Unanswerable questions reach a top score
  up to 0.68 (p90), above the p10 of answerable ones (0.64). Refusal needs a reranker or the LLM (M4).

```bash
docker compose --profile ml up -d --wait   # + TEI embedding server (BGE-M3)
uv run lexeu index embed                   # cache misses only, on a Modal GPU, parity-checked
uv run lexeu index build                   # blue/green: new collection, atomic alias switch
uv run lexeu eval retrieval --mlflow       # metrics per category and language, run logged
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The CI job **Eval (retrieval gate)** rebuilds the index and fails the pipeline if quality
drops below [`eval/thresholds.yaml`](eval/thresholds.yaml). Design: [ADR 0005](docs/adr/0005-embeddings-batch-and-online.md).

## Retrieval (M3)

Every idea became a config in [`eval/experiments.yaml`](eval/experiments.yaml), and one command
scores them all on the golden set (`uv run lexeu eval ablation --mlflow`, [full table](eval/baselines/m3_ablation.md)).

| Config | hit@1 | hit@5 | hit@10 | MRR@10 | p95 |
|---|---|---|---|---|---|
| dense (M2 baseline) | 0.530 | 0.881 | 0.940 | 0.687 | 25 ms |
| **dense + recital demotion** (production) | **0.722** | **0.914** | **0.954** | **0.809** | **28 ms** |
| hybrid dense + BM25 (RRF) | 0.430 | 0.848 | 0.914 | 0.606 | 30 ms |
| BM25 only | 0.272 | 0.649 | 0.788 | 0.437 | 28 ms |
| + bge-reranker-v2-m3, top 10 (GPU) | 0.755 | 0.934 | 0.954 | 0.836 | 0.6 s |
| + bge-reranker-v2-m3, top 30 (GPU) | 0.768 | 0.927 | 0.974 | 0.842 | 2.5 s |

- **One multiplication fixed the top-1 problem.** Scaling recital scores by 0.9 lets the article
  win close calls: +19 points hit@1 at zero latency. Recitals stay retrievable for context.
- **Hybrid search made things worse.** EU law repeats the same vocabulary everywhere ("Member
  States shall"), so BM25 adds noise, even at a 3:1 dense weight. The hypothesis from M2 was wrong,
  and the ablation says so.
- **Rerankers help, but not enough to pay for them yet.** +2 points hit@5 for 0.6-2.5 s on a GPU
  (or 5 s on CPU). Their real asset is a score that separates unanswerable questions; M4 decides
  whether refusals need it. Design: [ADR 0006](docs/adr/0006-retrieval-pipeline.md).

## Project layout

```
src/lexeu/
  api/          FastAPI app factory, middleware, routes
  core/         settings, logging
  infra/        Postgres registry (SQLAlchemy), S3 raw store, service clients
  ingestion/    corpus.toml, Cellar fetcher, OJ parser, chunker, sync pipeline
  retrieval/    embeddings (TEI client, cache), blue/green Qdrant index, dense retriever
  eval/         golden set schema, metrics, evaluation runner, quality gate
  jobs/         batch jobs on remote GPUs (Modal)
  commands/     `lexeu index` and `lexeu eval` sub-commands
  cli.py        `lexeu` command line
eval/           golden set, thresholds, baseline reports
migrations/     Alembic migrations
tests/          unit/ (fast, no I/O) and integration/ (real Postgres + S3)
docker/         container images
docs/adr/       architecture decision records
```

## License

MIT
