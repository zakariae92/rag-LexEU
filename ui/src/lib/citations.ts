/** Answer text -> plain segments and [n] citation markers, so markers can become links. */

export type Segment = { kind: "text"; text: string } | { kind: "cite"; n: number };

const MARKER = /\[(\d+(?:\s*,\s*\d+)*)\]/g; // [1] and [1, 3]

export function segment(text: string): Segment[] {
  const out: Segment[] = [];
  let last = 0;
  for (const match of text.matchAll(MARKER)) {
    const start = match.index ?? 0;
    if (start > last) out.push({ kind: "text", text: text.slice(last, start) });
    for (const n of (match[1] ?? "").split(",")) out.push({ kind: "cite", n: Number(n.trim()) });
    last = start + match[0].length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}
