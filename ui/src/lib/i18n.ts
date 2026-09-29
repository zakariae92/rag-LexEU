import type { Lang } from "./types";

export const STRINGS = {
  en: {
    title: "LexEU",
    tagline: "Answers on EU digital regulation, grounded in the official texts.",
    acts: "GDPR · AI Act · DORA · NIS2 · DSA · Data Act",
    placeholder: "Ask a question about EU regulation…",
    ask: "Ask",
    stop: "Stop",
    examples: "Try",
    sources: "Sources",
    searching: "Searching the regulations…",
    writing: "Writing the answer…",
    refused: "No answer in the sources",
    outOfScope: "Outside what I cover",
    newChat: "New conversation",
    understoodAs: "Understood as:",
    helpful: "Helpful",
    notHelpful: "Not helpful",
    thanks: "Thanks for the feedback",
    retry: "Retry",
    slow: "This is taking longer than usual…",
    busy: "The service is busy. Please retry in a few seconds.",
    rateLimited: "Too many questions right now. Please retry in a minute.",
    failed: "Something went wrong.",
    disclaimer:
      "Research demo, not legal advice. Answers cite the regulations' official EUR-Lex texts; check the linked provisions.",
    source: "Source code",
  },
  fr: {
    title: "LexEU",
    tagline: "Des réponses sur la réglementation numérique de l’UE, fondées sur les textes officiels.",
    acts: "RGPD · AI Act · DORA · NIS 2 · DSA · Data Act",
    placeholder: "Posez une question sur la réglementation européenne…",
    ask: "Envoyer",
    stop: "Arrêter",
    examples: "Exemples",
    sources: "Sources",
    searching: "Recherche dans les règlements…",
    writing: "Rédaction de la réponse…",
    refused: "Pas de réponse dans les sources",
    outOfScope: "Hors de mon périmètre",
    newChat: "Nouvelle conversation",
    understoodAs: "Question comprise :",
    helpful: "Utile",
    notHelpful: "Pas utile",
    thanks: "Merci pour votre retour",
    retry: "Réessayer",
    slow: "C’est plus long que d’habitude…",
    busy: "Le service est occupé. Réessayez dans quelques secondes.",
    rateLimited: "Trop de questions en ce moment. Réessayez dans une minute.",
    failed: "Une erreur s’est produite.",
    disclaimer:
      "Démo de recherche, pas un conseil juridique. Les réponses citent les textes officiels EUR-Lex ; vérifiez les dispositions liées.",
    source: "Code source",
  },
} satisfies Record<Lang, Record<string, string>>;

export type Strings = (typeof STRINGS)["en"];

export const EXAMPLES: Record<Lang, string[]> = {
  en: [
    "Within how many hours must a personal data breach be notified to the supervisory authority?",
    "Which AI practices are prohibited by the AI Act?",
    "What must a contract with an ICT third-party provider contain under DORA?",
  ],
  fr: [
    "Quelles sont les amendes maximales prévues par le RGPD ?",
    "Qu’est-ce qu’un système d’IA à haut risque selon l’AI Act ?",
    "Quelles entités sont concernées par la directive NIS 2 ?",
  ],
};
