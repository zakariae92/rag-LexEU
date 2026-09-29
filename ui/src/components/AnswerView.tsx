"use client";

import { useState } from "react";

import { segment } from "@/lib/citations";
import { sendFeedback } from "@/lib/client";
import type { Strings } from "@/lib/i18n";
import type { AskResponse, Source } from "@/lib/types";

export type Turn = {
  id: number;
  question: string;
  status: "searching" | "writing" | "done" | "error";
  answerId?: string;
  sources: Source[];
  draft: string;
  final?: AskResponse;
  error?: string;
};

export function AnswerView({ turn, t }: { turn: Turn; t: Strings }) {
  const text = turn.final?.answer ?? turn.draft;
  // Links for [n]: final citations when known, else the streamed sources (same numbering).
  const byN = new Map<number, Source>(turn.sources.map((s) => [s.n, s]));
  const cited = turn.final ? new Set(turn.final.citations.map((c) => c.n)) : null;
  const shown = cited ? turn.sources.filter((s) => cited.has(s.n)) : [];

  return (
    <article className="turn">
      <p className="question">{turn.question}</p>
      {turn.final?.standalone_question && (
        <p className="understood">
          {t.understoodAs} {turn.final.standalone_question}
        </p>
      )}

      {turn.status === "searching" && <p className="status pulse">{t.searching}</p>}
      {turn.status === "writing" && !turn.draft && <p className="status pulse">{t.writing}</p>}

      {turn.final?.refused && (
        <p className="badge">{turn.final.refusal_reason === "out_of_scope" ? t.outOfScope : t.refused}</p>
      )}
      {text && (
        <div className={`answer${turn.final?.refused ? " refused" : ""}`} aria-live="polite">
          {segment(text).map((s, i) =>
            s.kind === "text" ? (
              <span key={i}>{s.text}</span>
            ) : byN.has(s.n) ? (
              <a key={i} className="cite" href={byN.get(s.n)!.url} target="_blank" rel="noreferrer"
                 title={byN.get(s.n)!.citation}>
                {s.n}
              </a>
            ) : (
              <sup key={i}>[{s.n}]</sup>
            ),
          )}
          {turn.status === "writing" && <span className="caret" aria-hidden />}
        </div>
      )}

      {turn.status === "error" && <p className="error">{turn.error}</p>}

      {shown.length > 0 && (
        <section className="sources">
          <h3>{t.sources}</h3>
          <ol>
            {shown.map((s) => (
              <li key={s.n} value={s.n}>
                <a href={s.url} target="_blank" rel="noreferrer">{s.citation}</a>
              </li>
            ))}
          </ol>
        </section>
      )}

      {turn.final && (
        <footer className="meta">
          {!turn.final.refused && <Feedback answerId={turn.final.answer_id} t={t} />}
          <span>
            {((turn.final.timings_ms.total ?? 0) / 1000).toFixed(1)} s · {turn.final.usage.model.split("/").pop()}
          </span>
        </footer>
      )}
    </article>
  );
}

function Feedback({ answerId, t }: { answerId: string; t: Strings }) {
  const [sent, setSent] = useState<1 | -1 | null>(null);
  const rate = async (rating: 1 | -1) => {
    setSent(rating);
    if (!(await sendFeedback(answerId, rating))) setSent(null);
  };
  if (sent) return <span>{t.thanks}</span>;
  return (
    <span className="feedback">
      <button type="button" onClick={() => rate(1)} aria-label={t.helpful} title={t.helpful}>👍</button>
      <button type="button" onClick={() => rate(-1)} aria-label={t.notHelpful} title={t.notHelpful}>👎</button>
    </span>
  );
}
