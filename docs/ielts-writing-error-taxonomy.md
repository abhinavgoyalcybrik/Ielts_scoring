# IELTS Academic Writing — Complete Error Taxonomy

Reference rubric for an automated writing evaluator. Organised by the four official
criteria, with severity assigned by **reader impact**, not by grammar rule.

> **Note on sources:** IDP does not publish a "minor vs major" list. This taxonomy is
> derived from the public band descriptors (shared by IDP / British Council / Cambridge),
> where severity is determined by two things only: (1) does the error impede
> communication, and (2) how frequent is it.

---

## 1. Severity model

| Level | Definition | Effect on band |
|---|---|---|
| `band_limiting` | A structural absence or task failure | Hard-caps the criterion at a stated band |
| `major` | Reader must stop, re-read, or guess the meaning | Direct deduction |
| `systematic` | A `minor` error type repeated 4+ times | Weighted as `major` on that criterion |
| `minor` | Noticeable but meaning is fully clear | Only matters in aggregate |
| `stylistic` | Preference, not an error | **No deduction — report only, never "correct"** |

### Descriptor anchors the model should encode

| Descriptor wording | Band |
|---|---|
| "frequent error-free sentences", "a few errors" | 7 |
| "some errors … but they do not impede communication" | 6 |
| "errors can cause some difficulty for the reader" | 5 (cap) |
| "errors may severely distort the message" | 4 and below |

---

## 2. Task Response (Task 2) / Task Achievement (Task 1)

### 2.1 Band-limiting — Task 2

| Error | Cap |
|---|---|
| Under 250 words | TR 5 |
| Completely off-topic | TR 1–2 |
| Memorised essay largely unrelated to the prompt | TR 3 |
| Only one part of a multi-part question answered | TR 5 |
| No position stated in an opinion / agree-disagree essay | TR 5 |
| Position contradicts itself between intro and conclusion | TR 5–6 |
| Prompt copied verbatim as body content | Excluded from word count → may trigger under-length |

### 2.2 Band-limiting — Task 1

| Error | Cap |
|---|---|
| **No overview paragraph** (single most common band killer) | TA 5 |
| Under 150 words | TA 5 |
| Trend described backwards / data misread | TA 5 |
| Causes, opinions, or explanations invented (not in the data) | TA 5–6 |
| Second chart ignored when two are given | TA 5 |
| Process diagram: steps omitted or out of sequence | TA 5–6 |

### 2.3 Major

- Ideas asserted but never developed — no support, example, or consequence
- Uneven coverage of a two-sided question (one view 3 sentences, the other 15)
- Irrelevant or purely anecdotal examples
- Conclusion introduces brand-new arguments
- Task 1: no data/figures cited anywhere
- Task 1: every data point listed with no selection of key features
- Task 1: no comparison where the data clearly invites one
- Task 1: recommendations, predictions, or opinions added (off-task)

### 2.4 Minor

- Overview present but not signposted
- Thin one-line conclusion
- Slightly generic example
- Mild paragraph-length imbalance
- Word count well over the minimum (no penalty — time-management note only)

---

## 3. Coherence & Cohesion

### 3.1 Band-limiting

| Error | Cap |
|---|---|
| No paragraph breaks — single block of text | CC 5 |
| Paragraphs with no central idea; content scattered randomly | CC 5 |
| Every paragraph is one sentence | CC 5 |

### 3.2 Major

- **Mechanical linker overuse** — "Firstly / Moreover / Furthermore / In addition"
  heading nearly every sentence. The descriptors penalise *over-use* as explicitly
  as under-use.
- Wrong linker for the logical relation ("Therefore" where contrast is meant)
- Unclear referencing — "it / this / they" with no traceable antecedent
- No logical progression; ideas jump and the reader loses the thread
- Missing topic sentences
- Same idea restated in different words and presented as a new point
- One paragraph carrying two unrelated main ideas

### 3.3 Minor

- A single connector over-used ("However" four times)
- Topic sentence present but vague
- Occasional pronoun ambiguity that context resolves
- Missing transition between two otherwise clearly related paragraphs
- Heavy reliance on "and" / "but" to join clauses

---

## 4. Lexical Resource

### 4.1 Band-limiting

| Error | Cap |
|---|---|
| Vocabulary so limited/repetitive that meaning is obscured | LR 4 |
| Spelling so distorted words are unrecoverable | LR 4–5 |
| Extensive memorised phrasing | LR 4 |

### 4.2 Major

- Wrong word choice that changes meaning ("the government should *abolish* pollution")
- Wrong word form / part of speech ("the *develop* of technology", "*economical* growth")
- **Thesaurus misuse** — advanced word dropped into the wrong context. The classic
  Band 5 trap where reaching for "big words" actively lowers the score.
- Register violations in an academic essay:
  contractions (don't, can't) · slang · "etc." · "&" · "kids", "stuff"
  · direct reader address ("as you can see")
- Uncountable nouns systematically pluralised: *informations, advices, researches,
  equipments, knowledges*
- Prompt keyword repeated throughout with zero paraphrase
- Collocation errors that obscure meaning ("make a research", "give an exam")

### 4.3 Minor

- Recognisable spelling slips: *goverment, enviroment, beleive, occured, seperate,
  definately, recieve*
- Mild collocation awkwardness ("strong rain", "heavy traffic jam")
- Slightly informal but correct ("a lot of" for "many")
- Inconsistent British/American spelling **within one script**
- A word repeated 3–4 times where a synonym existed
- Imprecise but comprehensible word choice

---

## 5. Grammatical Range and Accuracy

### 5.1 Band-limiting

| Error | Cap |
|---|---|
| Only simple sentences; no subordination anywhere | GRA 5 |
| Errors in the majority of sentences, meaning frequently distorted | GRA 4 |
| No sentence boundaries / punctuation absent | GRA 4–5 |

### 5.2 Major

- Missing subject or verb ("Is very important that…", "Nowadays many people using…")
- Broken word order ("Nowadays is many people using internet")
- Fragments presented as sentences
- Run-on sentences with no boundaries
- Tense errors that change the timeline of the argument
- Task 1: dated past data described in present tense (or vice versa)
- Wrong form after a modal ("should to do", "can be improve")
- Malformed passive ("is discuss", "was happen")
- Broken conditionals ("If the government *will* ban…")
- Relative clause errors obscuring meaning ("the people *which*…", "students who *they* study")
- Comparative/superlative errors that **misstate Task 1 data** ("more higher", "the most highest")
- **Error clusters** — 3 or more errors inside a single sentence

### 5.3 Minor

- Article omission or insertion (a / an / the) — single instances
- Singular/plural on countable nouns
- Subject–verb agreement slips
- Prepositions: *discuss about, in the other hand, depend of, according to me,
  in nowadays, comprise of*
- Comma splices
- Missing comma after a fronted subordinate clause
- Capitalisation (internet/Internet, government/Government)
- Apostrophes (its / it's, student's / students')
- Awkward but grammatical complex structures

---

## 6. Escalation rules (compute, don't hard-code)

Severity is not a fixed property of an error type. Apply these at scoring time:

1. **Frequency escalation** — same `minor` type occurring **≥ 4 times** →
   reclassify as `systematic` and weight as `major` on that criterion.
   *One article error is invisible. Twenty article errors is a Band 5 GRA.*

2. **Cluster escalation** — **≥ 3 errors in one sentence** → all errors in that
   sentence escalate to `major`, regardless of individual type.

3. **Error-free sentence ratio** — this is the actual Band 7 test
   ("produces frequent error-free sentences"). Compute it explicitly:

   | Error-free sentences | GRA |
   |---|---|
   | < 30% | ≤ 5 |
   | ~50% | 6 |
   | ~70%+ | 7 |
   | ~90%+ **with** varied structures | 8 |

4. **Range gate** — accuracy alone cannot reach 7. If no complex structures are
   attempted, cap GRA at 6 even with zero errors.

5. **Band-limiting is a cap, not a deduction** — apply it to the criterion score
   directly. A missing Task 1 overview is not a sentence-level error and
   sentence-scanning logic will never find it. Check for it structurally.

---

## 7. Do NOT flag — false-positive suppression

Over-correction destroys evaluator credibility faster than under-correction.
None of the following are errors:

**Spelling & punctuation**
- British *and* American spelling are both accepted — never convert one to the other
- Oxford comma present or absent
- *Whilst, amongst, learnt, towards, analyse, organisation* — valid British forms

**Grammar myths**
- Starting a sentence with *And / But / Because* — `stylistic` at most
- Split infinitives
- Ending a sentence with a preposition
- Singular *they*
- Correctly-formed passive voice — never "correct" it to active

**Task conventions**
- First person ("I believe", "In my opinion") in Task 2 — explicitly acceptable
- Task 1 present tense when the chart carries no date
- Numerals used for figures in Task 1
- A one-sentence conclusion, if it genuinely summarises
- Word count above the minimum — no penalty exists

**Content**
- **Never flag an opinion as wrong.** IELTS assesses how an argument is expressed,
  not whether it is true, moral, or well-reasoned in the real world.
- Factual inaccuracy in a Task 2 example is not assessed
- Repeating a topic noun with no natural synonym (*government, technology, education*)
  is not a repetition penalty

**Most important**
- If a sentence is already correct, **return no error object.** Do not generate a
  "better version" to demonstrate activity. A Band 9 examiner-written answer must
  come back with an empty or near-empty mistakes array.

---

## 8. Suggested error object schema

```json
{
  "criterion": "GRA",
  "severity": "minor",
  "type": "article_omission",
  "original": "Government should invest in education.",
  "corrected": "The government should invest in education.",
  "explanation": "Singular countable noun needs a determiner.",
  "impedes_communication": false,
  "sentence_index": 12,
  "occurrence_count": 7,
  "escalated_to": "systematic"
}
```

Top-level additions worth carrying over from the speaking evaluator:

```json
{
  "band_limiting_findings": [
    {
      "criterion": "TA",
      "finding": "no_overview",
      "caps_criterion_at": 5,
      "evidence": "No paragraph summarises the main trend."
    }
  ],
  "systematic_errors": [
    { "type": "article_omission", "count": 7, "criterion": "GRA" }
  ],
  "minor_mistakes": [],
  "error_free_sentence_ratio": 0.42,
  "complex_structures_attempted": 5,
  "word_count": 268,
  "criterion_scores": { "TR": 6, "CC": 6, "LR": 6, "GRA": 5 },
  "overall_band": 6.0
}
```

Keep `minor_mistakes` in its own array, separate from band-affecting errors — the
same split already used in the speaking model. Candidates need to see them, but they
must not visibly drag the score unless escalation rule 1 fires.

---

## 9. L1 interference patterns — Hindi / Punjabi speakers

High-frequency, high-value patterns for the model to recognise and group as
`systematic` rather than reporting one at a time:

| Pattern | Example | Cause |
|---|---|---|
| Article omission / over-insertion | "Government should ban the plastic" | No articles in L1 |
| Present continuous for habitual action | "I am going to gym daily" | L1 aspect mapping |
| *since / from* confusion | "living here from five years" | Direct translation |
| *the* before abstract nouns | "The education is important" | Over-correction |
| Verb-final or shifted word order | "This problem we can solve easily" | L1 SOV order |
| Uncountables pluralised | "many informations" | L1 countability differs |
| *only* / *itself* as emphasis | "Today only I finished it" | Indian English idiom |
| Formulaic openers | "As we all know", "It is a fact that", "Nowadays" | Coaching-centre templates |
| *would* for future | "The government would take steps" | Modal mapping |

Report these as a single grouped finding with a count, not as N separate corrections —
that is what makes feedback actionable rather than overwhelming.
