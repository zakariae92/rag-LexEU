## Semantic cache threshold

36 paraphrases (should hit), 51 hand-written near-misses and 9376 pairs of distinct golden questions (must never hit).

| Threshold | Paraphrases served from cache | Near-miss hits | Golden-pair hits |
|---|---|---|---|
| 0.80 | 86.1% | 16 | 47 |
| 0.81 | 86.1% | 13 | 40 |
| 0.82 | 83.3% | 10 | 33 |
| 0.83 | 83.3% | 10 | 28 |
| 0.84 | 72.2% | 9 | 23 |
| 0.85 | 69.4% | 7 | 17 |
| 0.86 | 63.9% | 6 | 12 |
| 0.87 | 61.1% | 6 | 8 |
| 0.88 | 58.3% | 5 | 3 |
| 0.89 | 58.3% | 5 | 2 |
| 0.90 | 52.8% | 4 | 1 |
| 0.91 | 47.2% | 4 | 1 |
| 0.92 | 41.7% | 2 | 1 |
| 0.93 | 36.1% | 2 | 1 |
| 0.94 | 33.3% | 1 | 1 |
| 0.95 | 25.0% | 1 | 1 |
| 0.96 | 16.7% | 1 | 1 |
| 0.97 | 13.9% | 0 | 1 |
| 0.98 | 2.8% | 0 | 1 |
| 0.99 | 0.0% | 0 | 0 |

Most similar negative: 0.982 ("What are the maximum fines for essential entities under NIS2?" vs "What are the maximum fines for important entities under NIS2?").
Least similar paraphrase: 0.525 ("Les interfaces trompeuses sont-elles interdites par le règlement sur les services numériques ?" vs "Le DSA interdit-il les dark patterns ?").
Recommended threshold (max negative + 0.02): **none: do not cache**.

### Hardest negatives

- 0.982 [golden] What are the maximum fines for essential entities under NIS2? / What are the maximum fines for important entities under NIS2?
- 0.965 [pairs] What supervisory measures can authorities apply to essential entities? / What supervisory measures can authorities apply to important entities?
- 0.940 [pairs] From when does the GDPR apply? / When did the GDPR enter into force?
- 0.913 [pairs] Which AI practices are prohibited under the AI Act? / Which AI practices are allowed under the AI Act?
- 0.910 [pairs] Within how many hours must a controller notify a personal data breach to the supervisory authority? / Within how many hours must a processor notify a personal data breach to the controller?
- 0.896 [golden] How must financial entities classify ICT-related incidents? / Must financial entities report major ICT-related incidents, and to whom?
- 0.896 [pairs] What supervisory measures can authorities apply to essential entities? / What enforcement measures can authorities impose on essential entities?
- 0.884 [golden] Compare the obligations of management bodies under NIS2 and DORA. / Compare incident reporting obligations under DORA and NIS2.
- 0.876 [golden] What must the risk management system for high-risk AI systems involve? / What must the quality management system of a high-risk AI provider cover?
- 0.873 [golden] What are the obligations of providers of general-purpose AI models? / What additional obligations apply to providers of general-purpose AI models with systemic risk?

### Paraphrases that would miss (36)

- 0.525 Les interfaces trompeuses sont-elles interdites par le règlement sur les services numériques ? / Le DSA interdit-il les dark patterns ?
- 0.598 What backup and restoration policies must financial entities have? / What does DORA require for backups and data restoration?
- 0.709 What supervisory measures can authorities apply to essential entities? / Which supervisory powers do authorities have over essential entities under NIS2?
- 0.747 What rules apply to decisions based solely on automated processing, including profiling? / What does the GDPR say about fully automated decision-making and profiling?
- 0.755 When must an organisation designate a data protection officer? / In which cases is it mandatory to appoint a DPO?
- 0.816 What additional obligations apply to providers of general-purpose AI models with systemic risk? / What extra duties do providers of GPAI models with systemic risk have?
- 0.830 À partir de quel âge un enfant peut-il consentir seul au traitement de ses données par un service de la société de l'information ? / Quel est l'âge minimum pour qu'un mineur consente seul au traitement de ses données en ligne ?
- 0.833 Within how many hours must a controller notify a personal data breach to the supervisory authority? / How quickly must a data breach be notified to the data protection authority?
- 0.836 Under what conditions is a hosting provider exempt from liability? / When is a hosting service not liable for the information it stores?
- 0.839 What security measures does the GDPR require for processing personal data? / Which technical and organisational security measures must controllers implement under the GDPR?
- 0.848 What must cloud providers do to prevent unlawful international governmental access to non-personal data? / How must data processing service providers protect non-personal data against unlawful foreign government access?
- 0.853 Within how many hours must a controller notify a personal data breach to the supervisory authority? / What is the deadline to report a data breach to the supervisory authority?
- 0.855 What fines can be imposed for infringing the GDPR? / What is the maximum administrative fine under the GDPR?
- 0.869 What must management bodies of essential and important entities do under NIS2? / What are the responsibilities of management bodies under NIS2?
- 0.873 What is the rule on unfair contractual terms concerning data access imposed unilaterally on an enterprise? / When is a data-sharing contractual term unilaterally imposed on a company unfair under the Data Act?
- 0.891 Quel est le montant maximal de l'amende en cas de pratique d'IA interdite ? / Quelle est l'amende maximale pour une pratique d'intelligence artificielle interdite ?
- 0.898 From when do the AI Act's prohibitions on AI practices apply? / When do the bans on prohibited AI practices take effect?
- 0.906 How must online platforms treat notices from trusted flaggers? / How should platforms handle reports submitted by trusted flaggers?
- 0.906 What contractual provisions must ICT services agreements include under DORA? / Which clauses are mandatory in contracts with ICT third-party service providers under DORA?
- 0.913 What must online platforms accessible to minors do? / What are the obligations of online platforms towards minors?
- 0.919 Which AI practices are prohibited under the AI Act? / What uses of AI does the AI Act ban?
- 0.922 How does the GDPR define personal data? / What counts as personal data under the GDPR?
- 0.924 Who must carry out a fundamental rights impact assessment before deploying a high-risk AI system? / Which deployers have to perform a fundamental rights impact assessment for high-risk AI?
- 0.938 How long does a controller have to respond to a data subject request? / What is the deadline for a controller to answer a data subject's request?
- 0.943 From when does the GDPR apply? / What is the date of application of the GDPR?
- 0.946 How long does a controller have to respond to a data subject request? / Within what time limit must the controller reply to a request from a data subject?
- 0.946 Which financial entities fall within the scope of DORA? / To which financial entities does DORA apply?
- 0.956 Quelle est la définition des données à caractère personnel dans le RGPD ? / Comment le RGPD définit-il les données personnelles ?
- 0.957 What does human oversight of high-risk AI systems require? / What are the human oversight requirements for high-risk AI?
- 0.959 What are the lawful bases for processing personal data under the GDPR? / On what legal grounds can personal data be processed under the GDPR?
- 0.962 Les entités financières doivent-elles notifier les incidents majeurs liés aux TIC ? / Un incident majeur lié aux TIC doit-il être signalé par une entité financière ?
- 0.973 When must an organisation designate a data protection officer? / When is an organisation required to appoint a data protection officer?
- 0.976 How does the GDPR define personal data? / What is the GDPR definition of personal data?
- 0.976 What are the lawful bases for processing personal data under the GDPR? / Which legal bases make processing of personal data lawful according to the GDPR?
- 0.978 Which AI practices are prohibited under the AI Act? / Which artificial intelligence practices are forbidden by the AI Act?
- 0.985 How does the AI Act define an AI system? / What is the definition of an AI system in the AI Act?

## Variant: similarity + same retrieved sources

Hit only if the similarity passes the threshold AND retrieval returns the same sources (top 8, production config).

| Rule | Threshold | Paraphrases served | False hits |
|---|---|---|---|
| same top-1 source | 0.80 | 66.7% | 11 |
| same top-1 source | 0.85 | 50.0% | 7 |
| same top-1 source | 0.90 | 41.7% | 4 |
| same top-3 set | 0.80 | 44.4% | 5 |
| same top-3 set | 0.85 | 44.4% | 4 |
| same top-3 set | 0.90 | 38.9% | 2 |
| same top-3, same order | 0.80 | 22.2% | 2 |
| same top-3, same order | 0.85 | 22.2% | 2 |
| same top-3, same order | 0.90 | 16.7% | 1 |
| same top-5 set | 0.80 | 22.2% | 3 |
| same top-5 set | 0.85 | 22.2% | 1 |
| same top-5 set | 0.90 | 16.7% | 1 |

Remaining false hits share their sources: "From when does the GDPR apply?" and "When did the GDPR enter into force?" are both answered by Art. 99, with different dates; the maximum fines for essential and important entities are in the same article, with different amounts. **Decision: no similarity-based answer cache; exact match on the normalised question only (ADR 0008).**
