### stack-mock-llm-1s

| Users | Requests | Failures | Req/s | p50 | p95 | p99 | p95 retrieval | p95 generation |
|---|---|---|---|---|---|---|---|---|
| 1 | 18 | 0 | 0.32 | 1200 ms | 1600 ms | 1600 ms | 440 ms | 988 ms |
| 5 | 93 | 0 | 1.59 | 1200 ms | 1400 ms | 1600 ms | 443 ms | 988 ms |
| 10 | 185 | 0 | 3.11 | 1200 ms | 1600 ms | 2700 ms | 469 ms | 988 ms |
| 20 | 315 | 0 | 5.31 | 1500 ms | 3600 ms | 5200 ms | 3092 ms | 987 ms |
| 40 | 381 | 0 | 6.4 | 4000 ms | 6400 ms | 9100 ms | 4909 ms | 988 ms |
