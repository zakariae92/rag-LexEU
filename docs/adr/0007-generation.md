# 0007: Generation with cited sources, deterministic grounding checks and explicit refusals

**Status:** Accepted

## Context

The retriever (ADR 0006) returns the right provision in the top 5 for 91 % of questions. M4 turns
those provisions into an answer. A legal assistant has three requirements that decide the design:
every claim must point to its source, the system must say "I don't know" rather than guess, and it
must stay fast and cheap (target: p95 < 2 s end to end, cost per question tracked).

## Decision

**Pipeline:** question -> retrieval (top 8 provisions) -> small-to-big expansion -> prompt with
numbered sources -> LLM, JSON output -> grounding check -> answer with citations, or refusal.

- **LiteLLM** as the gateway: switching provider (Gemini, OpenAI, a local Ollama model) is a
  settings change. Every call reports tokens and cost from LiteLLM's price table, pinned with the
  package version (no network fetch), so costs are reproducible.
- **Structured output** (`{"answerable": bool, "answer": "... [n]"}`) instead of free text: refusal
  is a field the code can read, not a sentence to parse.
- **Deterministic grounding check on every request:**
  - citations to sources that were not given are removed;
  - an answer with no valid citation is never shown: it becomes a refusal ("ungrounded").
  The check cannot tell whether a source really supports a sentence; the LLM judge measures that
  offline.
- **Four refusal reasons, all logged and measured:** no sources, not in sources (the model's
  decision), ungrounded, invalid output. The refusal message is localised (EN/FR).
- **Small-to-big context:** the search runs on small chunks (precise), but each retrieved provision
  is given to the model whole (all its parts, up to 6,000 characters around the retrieved part).
  Without it, Art. 30(2) DORA, split into points (a)-(f) and (g)-(i), reached the model as half a
  list, and the model refused. Measured on the full golden set, same judge:

  | Sources | Correct | Partial | Faithful | False refusals | p95 | $ per 1,000 |
  |---|---|---|---|---|---|---|
  | retrieved parts only | 84.9 % | 22 | 98.6 % | 3.3 % | 1.50 s | 0.72 |
  | **whole provisions (small-to-big)** | **89.7 %** | **15** | **99.3 %** | **2.7 %** | 1.64 s | 0.79 |

  +5 points of fully correct answers for +150 ms and +10 % tokens.
- **Prompt v2** lets the model answer the part of a question the sources cover, and say what they
  do not cover, instead of refusing everything.
- **Model: `gemini-3.1-flash-lite`, minimal reasoning.** Measured on 33 stratified questions:

  | Generator | Refused unanswerable | False refusals | Generation p95 | $ per 1,000 questions |
  |---|---|---|---|---|
  | gemini-3.8-flash, low reasoning | 100 % | 9.7 % | 6.1 s | 2.14 |
  | gemini-3.8-flash, minimal | 100 % | 12.9 % | 3.6 s | 2.16 |
  | gemini-3.5-flash-lite, minimal | 100 % | 9.7 % | 1.7 s | 0.97 |
  | **gemini-3.1-flash-lite, minimal** | **100 %** | **6.5 %** | **1.6 s** | **0.73** |

  The larger model is not better at this task: the answer is in the sources, so reading and
  citing them does not need more reasoning, and reasoning is what makes 3.8-flash slow.

**Evaluation (`lexeu eval answers`):** refusal recall, false refusal rate, citation hit and
precision against the golden set, grounding hygiene, an **LLM judge** (correctness against the
reference answer, faithfulness to the cited sources), latency and cost. Results on the full golden
set (165 questions), production config, judge `gemini-3.8-flash`:

| Refused unanswerable | False refusals | Citation hit | Invalid citations | Correct | Correct or partial | Faithful | p95 | $ per 1,000 |
|---|---|---|---|---|---|---|---|---|
| 100 % | 2.7 % | 95.9 % | 0 % | 89.7 % | 100 % | 99.3 % | 1.64 s | 0.79 |

**Judge: `gemini-3.8-flash`**, stronger than the generator. Gemini 3.1 Pro was the first choice,
but its free tier allows about one judged run per day. The judge is a second signal: its verdicts
are biased (verbosity, same model family), so the deterministic metrics stay primary.

**CI answer gate:** the eval job runs the golden set end to end and fails below
`eval/thresholds.yaml` (`answers:`), including p95 <= 2 s and cost per question. An LLM response
cache, keyed by model, parameters and exact messages, makes reruns free when neither the prompt nor
the retrieved sources changed. Without the `GEMINI_API_KEY` secret, the step is skipped, not failed.

## Consequences

- Retrieval is no longer the whole latency budget: generation is about 95 % of it.
- A provider error (timeout, quota) returns a retryable 503 from the API, and is recorded as an
  error in the evaluation instead of aborting a 165-question run.
- Cost of a judged golden-set run: about $0.43 at paid-tier prices (free on the free tier).
- Caveats: the free tier of the Gemini API may use prompts to improve Google's products. It is fine
  for public legal texts and evaluation questions, but a production deployment with real user
  questions should use the paid tier or Vertex AI (EU data residency). The golden set has only 14
  unanswerable questions: refusal recall moves by 7 points per item.
