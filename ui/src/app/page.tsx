import { headers } from "next/headers";

import { Chat } from "@/components/Chat";
import type { Lang } from "@/lib/types";

export default async function Home() {
  // The interface language comes from the browser's preference, known at request time: no
  // client-side switch after hydration (no flash of English for French visitors).
  const accept = (await headers()).get("accept-language") ?? "";
  const lang: Lang = accept.toLowerCase().startsWith("fr") ? "fr" : "en";
  return <Chat initialLang={lang} />;
}
