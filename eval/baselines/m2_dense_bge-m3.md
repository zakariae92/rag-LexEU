<!-- M2 baseline, 2026-09-29: dense retrieval, BAAI/bge-m3 @5617a9f, index chunks_dbebb831400e, golden v1 (165 items, draft). -->
## Retrieval: dense BAAI/bge-m3, k=10

| Slice | n | hit@1 | hit@3 | hit@5 | hit@10 | recall@10 | mrr@10 | ndcg@10 |
|---|---|---|---|---|---|---|---|---|
| **overall** | 151 | 0.530 | 0.828 | 0.881 | 0.940 | 0.917 | 0.687 | 0.730 |
| category: cross_regulation | 13 | 0.385 | 0.692 | 0.769 | 0.846 | 0.769 | 0.566 | 0.538 |
| category: definition | 6 | 0.500 | 0.500 | 0.667 | 0.833 | 0.833 | 0.566 | 0.627 |
| category: false_premise | 5 | 0.600 | 1.000 | 1.000 | 1.000 | 1.000 | 0.800 | 0.852 |
| category: lookup | 57 | 0.509 | 0.807 | 0.842 | 0.930 | 0.921 | 0.661 | 0.720 |
| category: obligation | 60 | 0.567 | 0.917 | 0.983 | 1.000 | 0.967 | 0.743 | 0.788 |
| category: temporal | 10 | 0.600 | 0.700 | 0.700 | 0.800 | 0.800 | 0.664 | 0.696 |
| lang: en | 123 | 0.553 | 0.829 | 0.870 | 0.943 | 0.919 | 0.700 | 0.739 |
| lang: fr | 28 | 0.429 | 0.821 | 0.929 | 0.929 | 0.911 | 0.628 | 0.693 |

Latency per query: p50 241.6 ms, p95 578.6 ms.

Top-1 score, answerable vs unanswerable (can a threshold drive refusals?): top_score_answerable_median = 0.7272, top_score_answerable_p10 = 0.6354, top_score_unanswerable_median = 0.5743, top_score_unanswerable_p90 = 0.6842

### Misses at k=10 (9)

- `ai-022` expected 32024R1689:art_3; got Recital 13 AI Act, Art. 26(2) AI Act, Recital 93 AI Act
- `nis2-001` expected 32022L2555:art_2:p1; got Recital 8 NIS2, Recital 113 NIS2, Art. 2(2) NIS2
- `dsa-003` expected 32022R2065:art_8; got Recital 12 DSA, Recital 30 DSA, Recital 9 DSA
- `dsa-008` expected 32022R2065:art_25:p1; got Recital 67 DSA, Recital 38 Data Act, Considérant 55 DSA
- `data-005` expected 32023R2854:art_14, 32023R2854:art_15; got Art. 17(2) Data Act, point (i), Art. 21(4) Data Act, Art. 19(1) Data Act
- `time-006` expected 32022R2065:art_93:p2; got Art. 49(3) DSA, Art. 49(4) DSA, Art. 51(5) DSA
- `time-008` expected 32024R1689:art_113; got Considérant 97 RIA, Art. 51, par. 1, RIA, Considérant 112 RIA
- `cross-006` expected 32022L2555:art_20, 32022R2554:art_5:p2; got Art. 36(2) DORA, points (c)–(e), Art. 3 DORA, points (30)–(37), Recital 31 NIS2
- `cross-013` expected 32022R2065:art_2:p4; got Recital 7 Data Act, Art. 1(5) Data Act, Recital 10 AI Act
