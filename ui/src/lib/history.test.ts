import { describe, expect, it } from "vitest";

import { historyOf, MAX_EXCHANGES } from "./history";

describe("historyOf", () => {
  it("keeps completed exchanges, oldest first, as user/assistant turns", () => {
    expect(
      historyOf([
        { question: "GDPR breach deadline?", answer: "72 hours [1]." },
        { question: "And NIS2?" }, // still streaming, or failed: not part of the history
      ]),
    ).toEqual([
      { role: "user", content: "GDPR breach deadline?" },
      { role: "assistant", content: "72 hours [1]." },
    ]);
  });

  it("sends only the last exchanges, with long answers cut", () => {
    const many = Array.from({ length: 5 }, (_, i) => ({ question: `q${i}`, answer: "a".repeat(3000) }));
    const history = historyOf(many);
    expect(history).toHaveLength(2 * MAX_EXCHANGES);
    expect(history[0]).toEqual({ role: "user", content: "q2" });
    expect(history[1]?.content).toHaveLength(1500);
  });
});
