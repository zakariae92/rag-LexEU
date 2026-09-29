import { parseSse } from "./sse";
import type { Lang, StreamEvent } from "./types";

export class AskError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly retryAfterS: number | null = null,
  ) {
    super(message);
  }
}

/** Ask through the UI's own route handler, which adds the API key server-side. */
export async function* askStream(
  question: string,
  lang: Lang | null,
  signal: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const resp = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(lang ? { question, lang } : { question }),
    signal,
  });
  if (!resp.ok || !resp.body) {
    const retry = resp.headers.get("Retry-After");
    const detail = await resp
      .json()
      .then((b: { detail?: unknown }) => (typeof b.detail === "string" ? b.detail : null))
      .catch(() => null);
    throw new AskError(detail ?? `HTTP ${resp.status}`, resp.status, retry ? Number(retry) : null);
  }
  for await (const { event, data } of parseSse(resp.body)) {
    const body = JSON.parse(data);
    if (event === "sources") yield { type: "sources", answerId: body.answer_id, sources: body.sources };
    else if (event === "delta") yield { type: "delta", text: body.text };
    else if (event === "done") yield { type: "done", answer: body };
    else if (event === "error") yield { type: "error", detail: body.detail };
  }
}

export async function sendFeedback(answerId: string, rating: 1 | -1): Promise<boolean> {
  const resp = await fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ answer_id: answerId, rating }),
  });
  return resp.ok;
}
