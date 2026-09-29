"""Prometheus metrics: what dashboards and alerts are built on.

Labels stay low-cardinality (route templates, not raw paths; reasons, not questions): per-request
detail belongs in traces and in the answer log, not in metric labels.
"""

from prometheus_client import Counter, Histogram

from lexeu.generation.answer import Answer

# Buckets around the 2 s p95 objective, so the SLO line falls on a bucket boundary.
LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0)

HTTP_REQUESTS = Counter("lexeu_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram(
    "lexeu_http_request_duration_seconds", "HTTP request duration", ["route"],
    buckets=LATENCY_BUCKETS,
)  # fmt: skip
ANSWERS = Counter(
    "lexeu_answers_total",
    "Answers served",
    ["outcome", "reason", "cache", "lang"],  # outcome: answered | refused
)
ANSWER_LATENCY = Histogram(
    "lexeu_answer_stage_seconds",
    "Answer latency by stage (retrieval, generation, first_token, total)",
    ["stage", "cache"],
    buckets=LATENCY_BUCKETS,
)
LLM_TOKENS = Counter("lexeu_llm_tokens_total", "LLM tokens", ["model", "kind"])
LLM_COST = Counter("lexeu_llm_cost_usd_total", "LLM cost in USD", ["model"])
LLM_ERRORS = Counter("lexeu_llm_errors_total", "Failed answer generations", ["error"])
RATE_LIMITED = Counter("lexeu_rate_limited_total", "Requests refused by the rate limiter")
FEEDBACK = Counter("lexeu_feedback_total", "User feedback", ["rating"])


def record_answer(answer: Answer) -> None:
    cache = "hit" if answer.cached else "miss"
    ANSWERS.labels(
        outcome="refused" if answer.refused else "answered",
        reason=answer.refusal_reason or "none",
        cache=cache,
        lang=answer.lang,
    ).inc()
    for stage, ms in answer.timings_ms.items():
        if stage in ("retrieval", "generation", "first_token", "total"):
            ANSWER_LATENCY.labels(stage=stage, cache=cache).observe(ms / 1000)
    if not answer.cached:  # a cache hit calls no model
        LLM_TOKENS.labels(model=answer.model, kind="input").inc(answer.input_tokens)
        LLM_TOKENS.labels(model=answer.model, kind="output").inc(answer.output_tokens)
        LLM_COST.labels(model=answer.model).inc(answer.cost_usd)
