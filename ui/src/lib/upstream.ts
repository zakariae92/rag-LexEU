import "server-only"; // the API key must never reach a client bundle: importing this there fails

const API_URL = (process.env.LEXEU_API_URL ?? "http://127.0.0.1:8080").replace(/\/$/, "");

export function json(detail: string, status: number, headers: HeadersInit = {}): Response {
  return Response.json({ detail }, { status, headers });
}

/** Call the LexEU API with the UI's key. Returns the upstream response, or an error response. */
export async function callApi(path: string, body: unknown, signal: AbortSignal): Promise<Response> {
  const key = process.env.LEXEU_API_KEY;
  if (!key) return json("the UI is not configured (LEXEU_API_KEY is missing)", 503);
  try {
    return await fetch(`${API_URL}${path}`, {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal, // the visitor closing the tab cancels the answer upstream
      cache: "no-store",
    });
  } catch (err) {
    if (signal.aborted) return json("cancelled", 499);
    console.error("upstream_unreachable", path, err);
    return json("the answer service is unreachable, please retry", 502, { "Retry-After": "5" });
  }
}

/** Forward an upstream error with its status, message and Retry-After. */
export async function forwardError(upstream: Response): Promise<Response> {
  const detail = await upstream
    .json()
    .then((b: { detail?: unknown }) => (typeof b.detail === "string" ? b.detail : null))
    .catch(() => null);
  const retry = upstream.headers.get("Retry-After");
  return json(detail ?? `upstream error ${upstream.status}`, upstream.status, retry ? { "Retry-After": retry } : {});
}
