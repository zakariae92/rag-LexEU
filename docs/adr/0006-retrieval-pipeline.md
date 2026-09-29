# 0006: Retrieval pipeline chosen by ablation: dense + recital demotion, no online reranker (yet)

**Status:** Accepted (to be revisited after M4 measures refusal quality)

## Context

The M2 baseline (dense BGE-M3) found the right provision in the top 5 for 88 % of questions,
but ranked it first only 53 % of the time. The error analysis pointed at recitals: in 47 of the
71 top-1 misses, a recital (explanatory text, phrased like a question) outranked the article
that holds the rule. Three families of changes were measured on golden set v1 (151 questions with
a retrieval target) with `lexeu eval ablation`. Every run is logged in MLflow.

| Config | hit@1 | hit@5 | hit@10 | MRR@10 | p95 |
|---|---|---|---|---|---|
| dense (M2 baseline) | 0.530 | 0.881 | 0.940 | 0.687 | ~25 ms |
| **dense + recital demotion (x0.9)** | **0.722** | **0.914** | **0.954** | **0.809** | **~25 ms** |
| dense, recitals excluded | 0.729 | 0.914 | 0.947 | 0.812 | ~25 ms |
| hybrid dense + BM25 (RRF) | 0.430 | 0.848 | 0.914 | 0.606 | ~30 ms |
| BM25 only | 0.272 | 0.649 | 0.788 | 0.437 | ~30 ms |
| + bge-reranker-v2-m3, top 10 (GPU) | 0.755 | 0.934 | 0.954 | 0.836 | ~0.6 s |
| + bge-reranker-v2-m3, top 30 (GPU) | 0.768 | 0.927 | 0.974 | 0.842 | ~2.6 s |
| + mMiniLMv2 reranker, top 30 (GPU) | 0.788 | 0.920 | 0.960 | 0.841 | ~0.4 s (5.3 s on the dev CPU) |

## Decision

Production retrieval = **dense BGE-M3 + recital demotion (score x0.9), one slot per provision.**

- The demotion is the largest single gain (+19 points hit@1, +12 MRR) at zero latency or cost.
  Recitals stay retrievable (unlike exclusion), since they help interpret articles in answers.
- **Hybrid BM25 is rejected.** It lowers every metric, even with a 3:1 dense weight. EU legislation
  reuses the same vocabulary everywhere ("Member States shall", "competent authority"), so exact
  terms discriminate poorly, while BGE-M3 handles paraphrase. The BM25 vectors stay in the index
  (they are cheap) for later keyword-style queries such as "art 33 GDPR".
- **No online reranker for now.** Rerankers add +2 points of hit@5 (the passages the LLM will
  read) for 0.6-2.5 s on a GPU or 1.75-5 s on the dev CPU. Serving on Modal means either cold
  starts of 15-40 s (scale to zero) or a GPU billed 24/7, and user questions would leave the EU.
- Rerankers do have one clear advantage: **bge-reranker scores separate unanswerable questions**
  (answerable p10 0.80 vs unanswerable p90 0.67), which dense similarity cannot (0.64 vs 0.68).
  M4 will first measure refusals driven by the LLM and a grounding check; if they are not good
  enough, the reranker comes back as a refusal signal.

## Consequences

- The CI gate evaluates this configuration; thresholds were raised to its scores minus a margin.
- The reranker code, pinned models and Modal job remain, one config line away (`rerank:`).
- Caveats: the demotion factor (0.9) was set a priori, not tuned on the golden set. The golden set
  is written in natural language, which may under-represent keyword queries where BM25 shines. The
  unanswerable slice is small (14 items).
