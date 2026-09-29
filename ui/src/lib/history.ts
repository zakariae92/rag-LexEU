import type { HistoryTurn } from "./types";

export const MAX_EXCHANGES = 3; // what the API uses to understand a follow-up
const MAX_CHARS = 1500; // the rewrite needs an answer's topic, not all of it

type Exchange = { question: string; answer?: string };

/** The conversation so far, as the API expects it: completed exchanges only, most recent last. */
export function historyOf(exchanges: Exchange[]): HistoryTurn[] {
  return exchanges
    .filter((x): x is Required<Exchange> => typeof x.answer === "string" && x.answer.length > 0)
    .slice(-MAX_EXCHANGES)
    .flatMap((x) => [
      { role: "user" as const, content: x.question.slice(0, MAX_CHARS) },
      { role: "assistant" as const, content: x.answer.slice(0, MAX_CHARS) },
    ]);
}
