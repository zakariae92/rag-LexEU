"use client";

import { useEffect, useRef, useState } from "react";

import { AnswerView, type Turn } from "@/components/AnswerView";
import { AskError, askStream } from "@/lib/client";
import { EXAMPLES, STRINGS, type Strings } from "@/lib/i18n";
import type { Lang } from "@/lib/types";

export function Chat({ initialLang }: { initialLang: Lang }) {
  const [lang, setLang] = useState<Lang>(initialLang);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [slow, setSlow] = useState(false);
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const nextId = useRef(0);
  const bottom = useRef<HTMLDivElement>(null);
  const t = STRINGS[lang];

  useEffect(() => {
    // A block body: an effect must return nothing or a cleanup function (scrollIntoView now
    // returns a Promise in some browsers, which React would try to call on cleanup).
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  const update = (id: number, patch: (turn: Turn) => Partial<Turn>) =>
    setTurns((all) => all.map((x) => (x.id === id ? { ...x, ...patch(x) } : x)));

  async function ask(question: string) {
    question = question.trim();
    if (question.length < 3 || abort.current) return;
    const id = ++nextId.current;
    const controller = new AbortController();
    abort.current = controller;
    setBusy(true);
    setInput("");
    setTurns((all) => [...all, { id, question, status: "searching", sources: [], draft: "" }]);
    // No first event after 5 s: tell the visitor it is slower than usual rather than stuck.
    const coldTimer = setTimeout(() => setSlow(true), 5000);
    try {
      // The answer language follows the question; the UI language is only the interface's.
      for await (const ev of askStream(question, null, controller.signal)) {
        clearTimeout(coldTimer);
        setSlow(false);
        if (ev.type === "sources") update(id, () => ({ status: "writing", answerId: ev.answerId, sources: ev.sources }));
        else if (ev.type === "delta") update(id, (x) => ({ draft: x.draft + ev.text }));
        else if (ev.type === "done") update(id, () => ({ status: "done", final: ev.answer })); // authoritative
        else update(id, () => ({ status: "error", error: ev.detail }));
      }
    } catch (err) {
      if (!controller.signal.aborted) update(id, () => ({ status: "error", error: describe(err, t) }));
      else update(id, (x) => ({ status: x.draft ? "done" : "error", error: x.draft ? undefined : t.stop }));
    } finally {
      clearTimeout(coldTimer);
      setSlow(false);
      abort.current = null;
      setBusy(false);
    }
  }

  return (
    <main className="shell">
      <header className="top">
        <div>
          <h1>{t.title}</h1>
          <p className="tagline">{t.tagline}</p>
          <p className="acts">{t.acts}</p>
        </div>
        <div className="lang" role="group" aria-label="Language">
          {(["en", "fr"] as const).map((l) => (
            <button key={l} type="button" aria-pressed={lang === l} onClick={() => setLang(l)}>
              {l.toUpperCase()}
            </button>
          ))}
        </div>
      </header>

      <section className="thread">
        {turns.length === 0 && (
          <div className="examples">
            <p>{t.examples}</p>
            {EXAMPLES[lang].map((q) => (
              <button key={q} type="button" onClick={() => ask(q)}>{q}</button>
            ))}
          </div>
        )}
        {turns.map((turn) => (
          <AnswerView key={turn.id} turn={turn} t={t} />
        ))}
        {slow && <p className="status">{t.slow}</p>}
        <div ref={bottom} />
      </section>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          void ask(input);
        }}
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void ask(input);
            }
          }}
          placeholder={t.placeholder}
          maxLength={2000}
          rows={2}
          aria-label={t.placeholder}
        />
        {busy ? (
          <button type="button" onClick={() => abort.current?.abort()}>{t.stop}</button>
        ) : (
          <button type="submit" disabled={input.trim().length < 3}>{t.ask}</button>
        )}
      </form>

      <footer className="foot">
        <p>{t.disclaimer}</p>
        <a href="https://github.com/zakariae92/rag-LexEU" target="_blank" rel="noreferrer">{t.source}</a>
      </footer>
    </main>
  );
}

function describe(err: unknown, t: Strings): string {
  if (err instanceof AskError) {
    if (err.status === 429) return t.rateLimited;
    if (err.status === 503 || err.status === 502) return t.busy;
    return `${t.failed} (${err.message})`;
  }
  return t.failed;
}
