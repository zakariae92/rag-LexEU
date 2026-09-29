import { describe, expect, it } from "vitest";

import { parseSse, type SseEvent } from "./sse";

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(enc.encode(c));
      controller.close();
    },
  });
}

async function collect(chunks: string[]): Promise<SseEvent[]> {
  const out: SseEvent[] = [];
  for await (const ev of parseSse(streamOf(chunks))) out.push(ev);
  return out;
}

describe("parseSse", () => {
  it("parses the API's events in order", async () => {
    const body =
      'event: sources\ndata: {"n":1}\n\n' + 'event: delta\ndata: {"text":"72 h"}\n\n' + "event: done\ndata: {}\n\n";
    expect(await collect([body])).toEqual([
      { event: "sources", data: '{"n":1}' },
      { event: "delta", data: '{"text":"72 h"}' },
      { event: "done", data: "{}" },
    ]);
  });

  it("reassembles events split across network chunks, even inside a character", async () => {
    const body = 'event: delta\ndata: {"text":"délai"}\n\n';
    const bytes = new TextEncoder().encode(body);
    const split = Array.from(bytes, (b) => new Uint8Array([b])); // one byte per chunk
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        split.forEach((s) => c.enqueue(s));
        c.close();
      },
    });
    const out: SseEvent[] = [];
    for await (const ev of parseSse(stream)) out.push(ev);
    expect(out).toEqual([{ event: "delta", data: '{"text":"délai"}' }]);
  });

  it("ignores comments, handles CRLF and a final event without a blank line", async () => {
    expect(await collect([": keep-alive\r\n\r\n", "data: a\r\n", "data: b"])).toEqual([
      { event: "message", data: "a\nb" },
    ]);
  });
});
