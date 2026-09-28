# 0004: Ingest from the Cellar API and chunk along the legal structure

**Status:** Accepted

## Context

The corpus is six EU acts (GDPR, AI Act, DORA, NIS2, DSA, Data Act) in English and French.
Answers must cite provisions precisely ("Art. 6(1)(f) GDPR"). Retrieval quality depends first
on what a chunk contains.

Observations while exploring the source:

- `eur-lex.europa.eu` answers automated requests with `HTTP 202` and an empty body (bot challenge).
- The **Cellar REST API** of the EU Publications Office serves the Official Journal XHTML by
  content negotiation (`/resource/celex/{CELEX}`, `Accept-Language: eng|fra`), with ETags.
- That XHTML carries **ELI identifiers** (`rct_12`, `cpt_II.sct_1`, `art_6`, `006.001`,
  `anx_III`) that are **identical in every language version**.

## Decision

1. **Source:** the official Cellar API, with retries on 429/5xx, fail-fast on 4xx and on TLS
   errors, concurrency capped at 3, an identifying User-Agent, and the OS certificate store (truststore).
2. **Raw zone:** original bytes stored content-addressed in S3 (`raw/{celex}/{lang}/{sha256}.xhtml`).
3. **Structure-aware chunking** instead of fixed-size windows:
   - the unit is the smallest citable part: numbered paragraph, recital, unnumbered article body, annex;
   - a unit above `max_chars` (1800) is split **between points, never inside one**, and each part
     repeats the lead-in sentence ("…at least one of the following applies:");
   - each chunk carries a **breadcrumb header** (act > chapter > section > article > paragraph),
     prepended before embedding, and a **ready-made citation**.
4. **Stable IDs:** `chunk_id = {celex}:{lang}:{eli_id}:{unit}:{part}` and a language-neutral
   `provision_key = {celex}:{eli_id}:{unit}`. The eval set can reference provisions, and EN/FR
   versions align 1:1 (verified: 100% of 2,866 provisions on the full corpus).
5. **Incremental sync:** a document is reprocessed only if its SHA-256 or the `pipeline_version`
   (parser version + chunking parameters) changed. Chunks are replaced in one transaction. Acts
   removed from `corpus.toml` are deleted. Every run is recorded in `ingestion_runs`.

## Consequences

- Full corpus: 12 documents, ~6,100 chunks, ingested in ~4 s. A no-op re-sync takes ~1 s.
- Chunking parameters are part of the pipeline version, so ablations (M3) re-chunk automatically.
- Tied to the OJ XHTML format ("oj-convex"). The parser is versioned and covered by fixture tests,
  and it fails loudly (`ParseError`) on unexpected documents.
- Not handled yet: consolidated versions (amendments, corrigenda) and cross-reference extraction (M3).
