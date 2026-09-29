## Retrieval ablation (k=10, baseline = `dense`)

| Config | hit@1 | hit@5 | hit@10 | mrr@10 | ndcg@10 | hit@5 definition | hit@5 temporal | hit@5 cross_regulation | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| `dense` | 0.530 | 0.881 | 0.940 | 0.687 | 0.730 | 0.667 | 0.700 | 0.769 | 24.9 |
| `dense-lang` | 0.543 (+0.013) | 0.881 (+0.000) | 0.940 (+0.000) | 0.695 (+0.008) | 0.735 (+0.005) | 0.667 | 0.700 | 0.769 | 28.5 |
| `dense-recitals-demote` | 0.722 (+0.192) | 0.914 (+0.033) | 0.954 (+0.013) | 0.809 (+0.123) | 0.822 (+0.091) | 0.667 | 0.700 | 0.846 | 27.5 |
| `dense-recitals-exclude` | 0.729 (+0.199) | 0.914 (+0.033) | 0.947 (+0.007) | 0.812 (+0.126) | 0.823 (+0.092) | 0.833 | 0.700 | 0.846 | 24.3 |
| `hybrid-rrf` | 0.430 (-0.099) | 0.848 (-0.033) | 0.914 (-0.026) | 0.606 (-0.081) | 0.660 (-0.071) | 0.500 | 0.700 | 0.769 | 29.8 |
| `hybrid-dbsf` | 0.450 (-0.080) | 0.848 (-0.033) | 0.920 (-0.020) | 0.623 (-0.063) | 0.676 (-0.054) | 0.500 | 0.700 | 0.769 | 28.8 |
| `hybrid-rrf-demote` | 0.490 (-0.040) | 0.848 (-0.033) | 0.920 (-0.020) | 0.640 (-0.046) | 0.687 (-0.044) | 0.500 | 0.700 | 0.769 | 29.1 |
| `hybrid-rrf-exclude` | 0.642 (+0.113) | 0.907 (+0.026) | 0.954 (+0.013) | 0.758 (+0.072) | 0.783 (+0.053) | 0.667 | 0.700 | 0.923 | 28.6 |
| `sparse-only` | 0.272 (-0.258) | 0.649 (-0.232) | 0.788 (-0.152) | 0.437 (-0.250) | 0.503 (-0.228) | 0.333 | 0.300 | 0.538 | 28.1 |
| `hybrid-w3-demote` | 0.536 (+0.007) | 0.881 (+0.000) | 0.934 (-0.007) | 0.680 (-0.006) | 0.720 (-0.010) | 0.667 | 0.700 | 0.769 | 29.4 |
| `rerank-bge-m3` | 0.768 (+0.238) | 0.927 (+0.046) | 0.974 (+0.033) | 0.842 (+0.156) | 0.848 (+0.118) | 0.833 | 0.800 | 0.846 | 2460.1 |
| `rerank-minilm` | 0.788 (+0.258) | 0.920 (+0.040) | 0.960 (+0.020) | 0.841 (+0.154) | 0.846 (+0.116) | 0.833 | 0.800 | 0.846 | 422.6 |
| `rerank-bge-m3-top10` | 0.755 (+0.225) | 0.934 (+0.053) | 0.954 (+0.013) | 0.836 (+0.149) | 0.845 (+0.114) | 0.667 | 0.800 | 0.846 | 627.5 |
| `rerank-bge-m3-top20` | 0.755 (+0.225) | 0.920 (+0.040) | 0.967 (+0.026) | 0.829 (+0.143) | 0.839 (+0.108) | 0.667 | 0.800 | 0.769 | 1305.7 |

- `dense`: M2 baseline: BGE-M3 dense, both languages, recitals kept
- `dense-lang`: only chunks in the question's language
- `dense-recitals-demote`: recital scores x0.9 so articles win close calls
- `dense-recitals-exclude`: recitals never returned
- `hybrid-rrf`: dense + BM25 (EN/FR stemming), reciprocal rank fusion
- `hybrid-dbsf`: dense + BM25, distribution-based score fusion
- `hybrid-rrf-demote`: hybrid RRF + recital demotion
- `hybrid-rrf-exclude`: hybrid RRF without recitals
- `sparse-only`: BM25 alone, to understand why the fusion hurts
- `hybrid-w3-demote`: weighted RRF (dense counts 3x BM25) + recital demotion
- `rerank-bge-m3`: dense + recital demotion, top 30 reranked by bge-reranker-v2-m3 (568M)
- `rerank-minilm`: same, reranked by mmarco-mMiniLMv2 (118M, CPU-sized)
- `rerank-bge-m3-top10`: bge-reranker-v2-m3 on the top 10 only (latency vs quality)
- `rerank-bge-m3-top20`: bge-reranker-v2-m3 on the top 20
