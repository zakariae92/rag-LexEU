import { callApi, forwardError, json } from "@/lib/upstream";

export const dynamic = "force-dynamic";

export async function POST(req: Request): Promise<Response> {
  const body: unknown = await req.json().catch(() => null);
  const { answer_id, rating } = (body ?? {}) as { answer_id?: unknown; rating?: unknown };
  if (typeof answer_id !== "string" || !/^[0-9a-f]{32}$/.test(answer_id) || (rating !== 1 && rating !== -1)) {
    return json("answer_id and a rating of 1 or -1 are required", 422);
  }
  const upstream = await callApi("/v1/feedback", { answer_id, rating }, req.signal);
  return upstream.ok ? new Response(null, { status: 204 }) : forwardError(upstream);
}
