# Competency Scoring Issues and Limitations

## Purpose of This Note
This document records some of the practical issues encountered while building the AES-inspired competency scoring MVP. It is intended as implementation evidence and as a useful source for the project methodology, evaluation, and limitations sections.

## Current Issue: Competency Scores Are Too Low

The main issue currently observed in the scoring output is that the competency scores are consistently too low to be useful as a realistic competency signal.

### What the current results show
The current evaluation output indicates:

- all scored rows fall into the `low` band,
- the low-band ratio is `1.0`,
- average similarity values are very small,
- after multiplying similarity by `100`, the final competency scores still remain very low.

In practical terms, the model is technically running correctly, but the score scale is not producing meaningful-looking competency outputs.

## Why This Is Happening

The problem appears to be mainly in the scoring design rather than in the implementation code itself.

### 1. Whole-description scoring is too noisy
The current scorer compares:

- one full employee description
- against one reference text for one skill

using TF-IDF cosine similarity.

That usually leads to low similarity values because the full description contains many unrelated words. Even if one sentence in the description provides strong evidence for a skill, that evidence gets diluted by all the other content in the description.

### 2. Raw cosine similarity is being treated too directly as competency
The current formula is:

`competency_score = similarity * 100`

This is simple and transparent, but it is also harsh. A cosine similarity of `0.08` may actually represent useful textual overlap in a sparse TF-IDF space, yet when multiplied by `100` it becomes a score of only `8`, which looks poor in human terms.

### 3. The current preprocessing is minimal
The text preprocessing currently only does:

- lowercasing,
- punctuation removal,
- whitespace normalization.

This keeps the MVP simple, but it also means the model is not doing any stronger normalization such as:

- stop-word removal,
- stemming,
- lemmatization,
- phrase expansion.

As a result, lexical variation still weakens the matching signal.

### 4. Reference descriptions are sometimes too narrow or too generic
The configured reference texts are useful as an MVP starting point, but they are still limited.

Two specific problems follow from this:

- some skill references do not include enough aliases or evidence phrases,
- any unconfigured skill falls back to a generic template.

If the employee text uses different wording from the reference wording, cosine similarity stays weak even when the underlying meaning is relevant.

## Core Interpretation
The current model is effectively measuring lexical overlap between:

- long, noisy employee text
- and short skill reference descriptions

and then interpreting that raw overlap as competency.

That design produces very low scores even when the pipeline itself is behaving as expected.

## Best Improvement Options

### 1. Score against chunks instead of the whole description
Instead of comparing one skill against the full employee description, a better MVP approach would be:

- split the description into sentences or short chunks,
- score each chunk against the skill reference,
- keep the best chunk score.

This would reduce dilution from unrelated text and make the scoring more sensitive to the strongest available evidence.

### 2. Move to hybrid scoring instead of pure TF-IDF
A stronger MVP design would combine two types of signals:

- direct evidence of exact or alias matches,
- supporting semantic similarity from TF-IDF.

A reasonable starting formula would be:

`final_score = 0.6 * exact_or_alias_match + 0.4 * best_chunk_similarity`

This is stronger because:

- direct mention of the skill should count heavily,
- semantic overlap should still contribute,
- the final result becomes easier to justify in the report.

### 3. Calibrate the score scale
The current scaling method is too literal. Instead of treating `cosine * 100` as a final competency score, the system should use calibrated evidence bands.

One possible interpretation would be:

- `0.00 - 0.05` = low evidence
- `0.05 - 0.12` = moderate evidence
- `0.12+` = strong evidence

Alternatively, the score could be rescaled based on the observed distribution in the actual dataset rather than treated as a direct percentage.

### 4. Improve vectorizer settings
The TF-IDF configuration can also be improved. Stronger defaults would likely include:

- `stop_words="english"`
- `ngram_range=(1, 2)` or `ngram_range=(1, 3)`
- `sublinear_tf=True`

These changes would not solve the design problem on their own, but they could improve the quality of the supporting similarity signal.

### 5. Expand reference texts with aliases and evidence phrases
Each skill reference should ideally include:

- canonical skill name,
- aliases,
- closely related phrases,
- realistic evidence phrases seen in employee descriptions.

For example, `SQL` should not rely only on the exact string `SQL`. It may also need references to:

- database querying,
- joins,
- reporting queries,
- relational databases,
- data extraction.

That would make the scorer less brittle.

## Why Hybrid Scoring Is More Defensible

The proposed hybrid formula:

`final_score = 0.6 * exact_or_alias_match + 0.4 * best_chunk_similarity`

is methodologically stronger because the two parts do different jobs.

### Exact or alias match answers:
Did the employee text explicitly mention the skill or a very close synonym?

### Best chunk similarity answers:
Is there a sentence whose wording strongly resembles the ideal evidence for that skill, even if the exact label is not present?

This is a better balance because:

- exact mention is highly transparent and easy to defend,
- similarity allows some flexibility when employees use indirect wording,
- chunking reduces the dilution problem caused by long descriptions.

## Example

If the skill is `SQL` and the description says:

`Built automated dashboards and queried customer data using SQL and Power BI.`

Then:

- exact match for `SQL` should contribute strongly,
- the best sentence should also have good similarity,
- the final score should be meaningfully high.

But if the description says:

`Analysed customer records and extracted data from relational databases.`

Then:

- there may be no exact `SQL` mention,
- but chunk similarity could still be moderate,
- so the final score should be moderate rather than zero.

That is more realistic than the current whole-text TF-IDF approach.

## Summary
The current low scores do not primarily mean the implementation is broken. They indicate that the current scoring logic is too dependent on raw lexical overlap between long employee descriptions and relatively narrow reference texts.

The most important next improvements are:

1. compare skills against chunks rather than full descriptions,
2. combine exact or alias evidence with similarity,
3. calibrate score thresholds properly,
4. improve TF-IDF settings,
5. expand reference descriptions with aliases and evidence phrases.

This issue is therefore best framed as a model-design limitation in the current MVP rather than a software defect.
