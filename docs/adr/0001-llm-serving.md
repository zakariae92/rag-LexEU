# 0001: Hybrid LLM serving behind a gateway

**Status:** Accepted

## Context

The assistant must answer legal questions faithfully and with citations, in English and French.
That calls for a strong model (≥ 30B parameters, or a frontier API model). The development
machine has no CUDA GPU and 16 GB of RAM: running a model this size locally on CPU gives
1–3 tokens/s, far from the p95 < 2 s target. Open-weight models remain a goal, both for data
control and to demonstrate model serving.

## Decision

- **Self-hosted open weights:** Qwen3-32B (Mistral Small as the alternative) served with **vLLM**
  on **Modal** serverless GPUs, which scale to zero when idle.
- **API models:** **Gemini**, a fast tier by default and a Pro tier for complex queries.
  Exact model versions are pinned in config.
- **Gateway:** all calls go through **LiteLLM**, which gives one OpenAI-compatible interface,
  automatic fallback (vLLM cold start → Gemini), per-request cost tracking and budgets.
- **Evaluation judge:** always from a different model family than the generator, to avoid
  self-preference bias.

## Consequences

- Changing models is a config change, and every model is scored against the same eval set.
- Cold starts on Modal (~30–60 s) are hidden by the Gemini fallback. The fallback rate is tracked.
- Two providers to operate. LiteLLM centralises keys, retries and cost accounting.
- Rejected: Ollama for serving (no continuous batching, a dev tool); an always-on rented GPU (cost).
