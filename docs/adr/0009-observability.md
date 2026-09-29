# 0009: Observability: metrics, traces, tested alerts, and load-tested capacity

**Status:** Accepted

## Context

M4 and M5 measured quality and latency offline, one question at a time. In production the
questions are: is the service meeting its objective right now (p95 < 2 s, few refusals, bounded
cost)? When it is not, which stage is at fault? And how many users can one deployment serve?

## Decision

**Three signals, each with one job.**

- **Metrics (Prometheus, `/metrics`)** for dashboards and alerts: HTTP traffic by route template,
  answers by outcome, refusal reason, cache and language, latency per stage (retrieval, first
  token, generation, total) in buckets around the 2 s objective, LLM tokens, cost and errors,
  rate-limit refusals, load shedding, feedback. Labels stay low-cardinality: per-request detail
  belongs in traces and in the answer log.
- **Traces (OpenTelemetry)** to find which stage failed or slowed down: one span per stage
  (query embedding, Qdrant search, small-to-big expansion, generation, cache lookup), exported
  over OTLP to Jaeger locally and to **Langfuse Cloud** (EU region) for LLM-level inspection
  (prompt, sources, answer, tokens, cost). Instrumenting with the standard (GenAI semantic
  conventions) instead of a vendor SDK keeps the backend a configuration choice. Langfuse is not
  self-hosted: its stack (ClickHouse, workers) needs 2-3 GB of RAM that the development machine
  does not have next to the embedding server. `TRACING__CAPTURE_CONTENT=false` keeps user text out
  of traces when questions may contain personal data.
- **Logs (structlog, JSON)** for events, with the request id that joins them to traces.

**Alerts on symptoms, unit-tested.** Prometheus rules fire on what users and the budget feel:
p95 above 2 s for 10 minutes, a refusal spike (with a minimum traffic), generation errors above
5 %, 5xx above 5 %, no API instance up, more than $1 of LLM spend in an hour, and more than 30 % of
negative feedback in a day. Each rule has a `promtool test rules` case (it fires, it stays silent,
its message), run by a CI job, so a broken rule is caught before an incident needs it. A Grafana
dashboard, provisioned as code, shows the same signals.

**Traces paid off immediately.** A first request took 13.8 s. Its trace showed 11.3 s in
retrieval and, with finer spans, in the query embedding: the idle TEI container had been paged out
(1.4 GB of free RAM on the host). The API now warms the embedding server at startup.

**Capacity, load-tested** (Locust, golden-set questions, 1-3 s think time, answer cache off). To
measure our stack and not the provider's rate limits, the model is replaced by a simulated one
answering in 1 s (`GENERATION__MOCK_LATENCY_MS`, never enabled in production):

| Users | Answers/s | p50 | p95 | p99 | p95 retrieval |
|---|---|---|---|---|---|
| 1 | 0.3 | 1.2 s | 1.6 s | 1.6 s | 0.4 s |
| 10 | 3.1 | 1.2 s | 1.6 s | 2.7 s | 0.5 s |
| 20 | 5.3 | 1.5 s | 3.6 s | 5.2 s | 3.1 s |
| 40 | 6.4 | 4.0 s | 6.4 s | 9.1 s | 4.9 s |

One deployment meets the objective up to about 10 concurrent users and saturates at about
6 answers per second. Span timings at 40 users locate the bottleneck: query embedding on the CPU
embedding server (p50 2.2 s), while Qdrant stays near 0.1 s and generation at its 1 s. With the
real model (`gemini-3.1-flash-lite`, 1-2 users, 54 questions), p95 is 1.7-2.0 s with no errors and
$0.00085 per answer.

**Admission control.** Past saturation every request queues behind the embedding server and all
users get slow answers. A bound of 8 answers in flight, with a 1 s queue, sheds the excess with a
fast `503 Retry-After: 2`. At 40 users (about twice the capacity): p95 2.6 s instead of 6.4 s and
p99 3.1 s instead of 9.1 s for admitted requests, with 47 % of requests shed. It does not add
capacity; it keeps the service usable for the users it admits.

## Consequences

- Scaling past about 6 answers/s means scaling query embedding: a GPU-backed TEI (about 10 ms per
  query instead of about 100 ms), or several CPU replicas behind a load balancer. The rest of the
  stack is far from its limits. `max_inflight_answers` must then be raised with the capacity.
- Latency numbers come from a laptop shared with Docker and the load generator: they are orders
  of magnitude for capacity planning, not benchmarks. Retrieval percentiles from Prometheus are
  interpolated within histogram buckets (0.25, 0.5, 0.75 s...).
- Grafana reads Prometheus only: Jaeger v2 serves just its v3 query API, which Grafana's Jaeger
  datasource does not read. The dashboard links to the Jaeger UI instead.
- Langfuse needs the user's own project keys (`TRACING__LANGFUSE_PUBLIC_KEY`/`_SECRET_KEY`).
