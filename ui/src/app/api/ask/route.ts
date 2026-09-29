// Backend-for-frontend: the browser calls this route, which calls the API with the UI's key.
// The key stays on the server, and the event stream is passed through as it arrives.

import { callApi, forwardError, json } from "@/lib/upstream";

export const dynamic = "force-dynamic";

export async function POST(req: Request): Promise<Response> {
  const body: unknown = await req.json().catch(() => null);
  const { question, lang, history = [] } = (body ?? {}) as {
    question?: unknown;
    lang?: unknown;
    history?: unknown;
  };
  if (typeof question !== "string" || !question.trim() || question.length > 2000) {
    return json("a message of 1 to 2000 characters is required", 422);
  }
  if (lang !== undefined && lang !== "en" && lang !== "fr") return json("lang must be en or fr", 422);
  if (!isHistory(history)) return json("history must be at most 20 user/assistant turns", 422);

  const upstream = await callApi(
    "/v1/ask/stream",
    { question: question.trim(), lang, history },
    req.signal,
  );
  if (!upstream.ok || !upstream.body) return upstream.ok ? json("empty stream", 502) : forwardError(upstream);

  return new Response(upstream.body, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "X-Accel-Buffering": "no",
    },
  });
}

function isHistory(value: unknown): boolean {
  return (
    Array.isArray(value) &&
    value.length <= 20 &&
    value.every(
      (t: { role?: unknown; content?: unknown }) =>
        (t?.role === "user" || t?.role === "assistant") &&
        typeof t.content === "string" &&
        t.content.length > 0 &&
        t.content.length <= 8000,
    )
  );
}
