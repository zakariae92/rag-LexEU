import { describe, expect, it } from "vitest";

import { segment } from "./citations";

describe("segment", () => {
  it("splits text around citation markers", () => {
    expect(segment("Within 72 hours [1]. Also [2, 3]")).toEqual([
      { kind: "text", text: "Within 72 hours " },
      { kind: "cite", n: 1 },
      { kind: "text", text: ". Also " },
      { kind: "cite", n: 2 },
      { kind: "cite", n: 3 },
    ]);
  });

  it("leaves text without markers, and non-numeric brackets, alone", () => {
    expect(segment("Article 5 [a] applies")).toEqual([{ kind: "text", text: "Article 5 [a] applies" }]);
    expect(segment("")).toEqual([]);
  });
});
