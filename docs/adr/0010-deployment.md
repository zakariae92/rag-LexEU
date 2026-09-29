# 0010: Deployment: one ARM VM, a compose stack behind Caddy, arm64 images, gated continuous deployment

**Status:** Accepted

## Context

The demo must be public (a URL recruiters can open), cost nothing to keep running, and show how a
release reaches production. The stack needs about 6 GB of RAM (the embedding server alone about
1.5-3 GB) and a CPU-bound embedding stage (ADR 0009).

## Decision

**One Oracle Cloud Always Free VM**: `VM.Standard.A1.Flex`, 4 Ampere (ARM64) cores, 24 GB of RAM,
free with no time limit. Serverless (Cloud Run with managed Qdrant, Postgres and Redis) was designed
first; the provider account could not be opened, and it would have added 20-40 s cold starts on a
scale-to-zero embedding server. A VM running the same compose stack as development keeps
production close to what is tested, and is always warm.

**The stack** ([`deploy/compose.prod.yml`](../../deploy/compose.prod.yml)): Caddy is the only
service with published ports (80, 443); it obtains and renews the Let's Encrypt certificate, sends
`/v1/*` and `/docs` to the API, everything else to the UI, and answers 404 for `/metrics`. The
event stream is excluded from compression so tokens reach the browser as they are written. A
one-shot `migrate` service runs Alembic before the API starts. Each service receives only the
variables it needs, and has a memory limit.

**An ONNX embedding server instead of TEI on ARM.** TEI publishes x86 CPU images only. TEI's CPU
path is the model's ONNX export under onnxruntime, then pooling and normalisation; a 150-line
server does the same with the same files and the same API (`/embed`, `/health`). Measured against
TEI on 48 golden-set questions and 16 indexed passages: cosine 1.000000, largest absolute
difference 1.9e-6, so the index is reused without re-embedding. It uses about 1.5 GB of RAM instead
of 3 GB. It runs one batch at a time over all cores; TEI's dynamic batching of concurrent requests
is not reproduced (to be measured by the load test on the VM).

**The UI holds its API key server-side.** The browser calls the Next.js route handlers, which call
the API with a key that never reaches a client bundle (`server-only`). All visitors share that
key's rate limit: it is the ceiling on the demo's LLM spend.

**Continuous deployment, gated.** On `main`, arm64 images are built on native ARM runners (no QEMU
emulation) and pushed to GHCR, tagged by commit, only after unit, integration and smoke tests, the
UI checks, and the retrieval and answer quality gates. The deploy job then connects as a dedicated
`deploy` user (its own key, revocable alone; the VM's host key pinned against interception), sets
the commit tag, pulls, and waits for every health check before checking the public URL. A rollback
is the previous tag.

## Consequences

- A single VM is a single point of failure: no redundancy, and maintenance means downtime. For a
  demo that trade-off buys a free, always-warm service; a production service would add a second
  instance behind a load balancer, managed databases and automated backups.
- Postgres backups are manual (`pg_dump`, see [docs/deploy.md](../deploy.md)); the index can be
  restored from its snapshot or rebuilt from the corpus.
- The VM itself was created in the console. Describing it in Terraform (and importing the existing
  resources) is the next step for reproducibility.
- Images are arm64 only: they match the VM; development builds its own images locally.
