## Answers: gemini/gemini-3.1-flash-lite

Model `gemini/gemini-3.1-flash-lite`, judge `gemini/gemini-3.8-flash`, 165 questions.

| Slice | n | refuse (unanswerable) | false refusal | retrieval hit | citation hit | citation precision | correct | correct+partial | faithful |
|---|---|---|---|---|---|---|---|---|---|
| **overall** | 163 | 1.000 | 0.027 | 0.953 | 0.959 | 0.564 | 0.897 | 1.000 | 0.993 |
| category: cross_regulation | 13 | - | 0.077 | 0.846 | 0.833 | 0.465 | 0.750 | 1.000 | 1.000 |
| category: definition | 6 | - | 0.000 | 0.833 | 0.833 | 0.750 | 1.000 | 1.000 | 1.000 |
| category: false_premise | 7 | 1.000 | 0.400 | 1.000 | 1.000 | 0.567 | 1.000 | 1.000 | 1.000 |
| category: lookup | 57 | - | 0.000 | 0.965 | 0.965 | 0.539 | 0.895 | 1.000 | 0.983 |
| category: obligation | 58 | - | 0.000 | 1.000 | 1.000 | 0.557 | 0.931 | 1.000 | 1.000 |
| category: temporal | 10 | - | 0.100 | 0.800 | 0.889 | 0.778 | 0.778 | 1.000 | 1.000 |
| category: unanswerable | 12 | 1.000 | - | - | - | - | - | - | - |
| lang: en | 132 | 1.000 | 0.025 | 0.959 | 0.966 | 0.579 | 0.907 | 1.000 | 0.992 |
| lang: fr | 31 | 1.000 | 0.036 | 0.929 | 0.926 | 0.498 | 0.852 | 1.000 | 1.000 |

Latency: end to end p50 1125 ms, **p95 1642 ms** (retrieval p95 98 ms, generation p95 1582 ms).
Cost: **$0.00079 per question** ($0.79 per 1,000), 2372 input + 134 output tokens on average. Judge: $0.3002 total. Response cache hits: 0%.

### Refused although answerable (4)

- `time-006` From when does the Digital Services Act apply? (source retrieved: False): not_in_sources
- `cross-001` How do breach notification deadlines under the GDPR compare with incident reporting deadlines under NIS2? (source retrieved: True): not_in_sources
- `premise-005` Given that NIS2 caps fines for essential entities at 4 % of turnover, how are they calculated? (source retrieved: True): not_in_sources
- `premise-007` Pourquoi le DSA ne s'applique-t-il qu'aux plateformes de plus de 10 millions d'utilisateurs ? (source retrieved: True): not_in_sources

### Judged unfaithful (1)

- `data-004` What is the rule on unfair contractual terms concerning data access imposed unilaterally on an enterprise? (source retrieved: True): The answer accurately states the core rule from the reference answer and provides relevant details. However, it fails faithfulness because the claim that severable terms leave the rest of the contract binding is cited to [7] (Art. 13(8)) instead of [8] (Art. 13(7)).

### Errors (not scored) (2)

- `nis2-003` What cybersecurity risk-management measures must essential and important entities take?: 
- `data-007` What must a cloud switching contract include?: 
