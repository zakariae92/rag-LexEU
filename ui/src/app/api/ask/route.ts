// Backend-for-frontend: the browser calls this route, which calls the API with the UI's key.
// The key stays on the server, and the event stream is passed through as it arrives.

import { callApi, forwardError, json } from "@/lib/upstream";

export const dynamic = "force-dynamic";

export async function POST(req: Request): Promise<Response> {
  const body: unknown = await req.json().catch(() => null);
  const { question, lang } = (body ?? {}) as { question?: unknown; lang?: unknown };
  if (typeof question !== "string" || question.trim().length < 3 || question.length > 2000) {
    return json("a question of 3 to 2000 characters is required", 422);
  }
  if (lang !== undefined && lang !== "en" && lang !== "fr") return json("lang must be en or fr", 422);

  const upstream = await callApi("/v1/ask/stream", { question: question.trim(), lang }, req.signal);
  if (!upstream.ok || !upstream.body) return upstream.ok ? json("empty stream", 502) : forwardError(upstream);

  return new Response(upstream.body, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "X-Accel-Buffering": "no",
    },
  });
}
