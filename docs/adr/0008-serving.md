# 0008: Serving: API keys, rate limiting, streaming, answer log, and no semantic cache

**Status:** Accepted

## Context

M4 produced grounded answers in 1.6 s at p95. Serving them to real clients needs: knowing who
calls, limiting abuse and cost, showing text early, recording what was answered (audit, feedback,
monitoring), and not paying twice for the same question.

## Decision

**API keys** (`Authorization: Bearer lx_...`), managed with `lexeu keys create | list | revoke`.
Keys are random 256-bit tokens stored as SHA-256 hashes: a leaked table leaks no keys, and a fast
hash is enough because keys are not human passwords. Authentication is required by default;
`AUTH__REQUIRED=false` allows anonymous calls, limited per IP, for local exploration.

**Rate limiting per key, in Redis**, with a sliding-window counter: a fixed window would let a
client send twice its limit around the minute boundary. The check and the increment run in one
Lua script, so concurrent requests cannot both slip through (tested: 50 concurrent requests
against a limit of 10 let exactly 10 through), and the limit holds across API replicas. Refusals
are `429` with `Retry-After`; every response carries `X-RateLimit-Limit` and `-Remaining`.

**Streaming** (`POST /v1/ask/stream`, server-sent events): `sources` right after retrieval,
`delta` events while the model writes (the `answer` field is decoded incrementally from the JSON
output), then `done` with the same body as `/v1/ask`. The grounding check needs the complete text,
so `done` is authoritative: a draft that fails the check ends as a refusal and clients replace it.
Measured live: sources at ~0.5 s, first text at ~1.3 s, full answer at ~1.9 s.

**Answer log and feedback** (Postgres, migration 0003): every answer gets an id and is stored with
its citations, model, prompt version, tokens, cost, latencies and whether it came from the cache.
The row is written after the response is sent, and a failed write never fails the request.
`POST /v1/feedback` stores a rating (+1/-1) and an optional comment per answer.

**No semantic cache.** A semantic cache serves the stored answer of a *similar* question. We
measured it before building it (`lexeu eval cache`, [study](../../eval/baselines/m5_semantic_cache.md)):
36 paraphrases, 51 hand-written near-misses (same topic, different answer), and 9,376 pairs of
distinct golden questions, all embedded with BGE-M3.

| Threshold | Paraphrases served from cache | Wrong-question hits |
|---|---|---|
| 0.85 | 69 % | 24 |
| 0.90 | 53 % | 5 |
| 0.95 | 25 % | 2 |
| 0.97 | 14 % | 1 |

The most similar pair of *different* questions scores 0.982: "maximum fines for essential entities"
vs "... for important entities" (NIS2), which have different amounts. "Prohibited" vs "allowed" AI
practices score 0.913, a controller's vs a processor's breach deadline 0.910. Requiring the same
retrieved sources as well does not fix it: "From when does the GDPR apply?" and "When did it enter
into force?" retrieve the same article, which gives two different dates. Embeddings capture the
topic, not the legal detail that changes the answer.

Instead, an **exact-match cache** on the normalised question (Unicode form, case, quotes, spacing,
trailing punctuation), per answer language, in Redis with a 7-day TTL. Its key namespace hashes the
live index fingerprint, the model, the prompt version and the retrieval config, so any change that
could alter an answer invalidates the cache without a flush. Model glitches (invalid output,
ungrounded draft) are never cached. A Redis outage only costs a model call. Measured live: a repeat
takes 1.6 ms server-side instead of 2.7 s, for $0.

## Consequences

- The cache only absorbs exact repeats. Its hit rate will be measured on real traffic (M6), from
  the answer log's `cache_hit` column.
- `lexeu eval cache` stays as the test any future similarity cache must pass: a better embedding
  model, or a cross-encoder that checks that two questions are equivalent.
- The cache namespace is computed at startup: rebuilding the index requires a restart (which a
  blue/green deployment does anyway).
