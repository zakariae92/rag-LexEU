// The API's contract (src/lexeu/api/routes/ask.py). `done` carries the same body as /v1/ask.

export type Lang = "en" | "fr";

export type Source = {
  n: number;
  citation: string;
  provision_key: string;
  url: string;
};

export type Citation = Source & { lang: string };

export type AskResponse = {
  answer_id: string;
  answer: string;
  lang: Lang;
  refused: boolean;
  refusal_reason: string | null;
  citations: Citation[];
  usage: { model: string; input_tokens: number; output_tokens: number; cost_usd: number };
  timings_ms: Record<string, number>;
  prompt_version: string;
};

export type StreamEvent =
  | { type: "sources"; answerId: string; sources: Source[] }
  | { type: "delta"; text: string }
  | { type: "done"; answer: AskResponse }
  | { type: "error"; detail: string };
