# Competency Scoring MVP: Step-by-Step Reasoning

## Purpose of This Note
This document explains the implementation logic behind the AES-inspired competency scoring MVP. It is written as a structured design rationale for the project report and implementation evidence. It does not record private chain-of-thought. Instead, it explains the concrete engineering decisions taken at each step and why they were made.

## Step 1: Start from the project constraint
The first design decision was to accept the project constraint exactly as given:

- input fields must be limited to `talentlinkId`, `description`, and `skills`
- the system must not infer new skills
- the system must only score skills that are already listed for the employee

This immediately ruled out broader approaches such as skill discovery, knowledge graphs, or LLM-based extraction. The implementation therefore needed to be a controlled scoring pipeline, not an open-ended NLP system.

## Step 2: Convert the existing data into the target schema
The current project CSV did not already use the exact final field names:

- `TalentLink ID` instead of `talentlinkId`
- `Description` instead of `description`
- `skills` stored as a stringified Python list

Because of that, the loader was designed to normalize the current project output into the canonical schema used by the MVP. This keeps the rest of the pipeline clean and allows the methodology to present one stable contract even if the raw input naming varies.

## Step 3: Validate the schema before any scoring happens
Validation was placed at the beginning of the pipeline because every later stage depends on the three required fields being usable.

The loader therefore checks:

- whether the required columns exist,
- whether descriptions can be converted safely to text,
- whether the `skills` field can be converted into a proper list,
- whether rows with empty descriptions or empty skills should be excluded.

This was done first so that the pipeline fails early and predictably if the data is malformed.

## Step 4: Collapse repeated rows to one employee record
The CSV in the project is row-based and may contain multiple rows for the same employee. The competency scoring system, however, is conceptually employee-level.

For that reason, the loader aggregates repeated rows by `talentlinkId` and creates:

- one combined description per employee,
- one unique list of listed skills per employee.

This matches the project objective more closely because the model is meant to judge whether an employee description supports each listed skill, not to score each raw row independently.

## Step 5: Parse the skills field into a controlled list
The `skills` column can appear in more than one usable form:

- already as a Python list,
- as a stringified list like `['Python', 'SQL']`,
- or as comma-separated text.

The parser was designed to handle these cases explicitly, deduplicate repeated skills, and preserve order. This improves robustness without introducing complex parsing logic.

## Step 6: Use conservative text preprocessing
The preprocessing stage was intentionally kept simple. The goal was not to build a sophisticated linguistic pipeline, but to make the text more consistent for TF-IDF comparison.

The chosen rules were:

- lowercase text,
- remove punctuation and noise,
- normalize repeated whitespace,
- preserve useful skill evidence terms.

This is a deliberate MVP tradeoff. More aggressive preprocessing could remove evidence words that matter to the scoring task.

## Step 7: Use curated skill reference descriptions
The project specification required an AES-style comparison against an ideal reference. That means each skill must have a short benchmark description explaining what strong evidence for that skill looks like.

The implementation therefore uses:

- a configured dictionary for known skills,
- a default fallback template for unknown configured cases.

This keeps the system explainable:

- the employee description is the observed answer,
- the skill reference is the ideal benchmark answer.

## Step 8: Fit TF-IDF on both employee text and reference text
The vectorizer is not fit only on employee descriptions. It is fit on:

- the cleaned employee descriptions,
- the skill reference texts.

This decision matters because the vocabulary used in reference descriptions must also exist in the TF-IDF space. If the vectorizer ignored the reference side, the comparison would be weaker and less stable.

## Step 9: Score only listed employee-skill pairs
For each employee, the scorer loops only through the skills already listed for that employee.

This is central to the project:

- it does not search the description for all possible skills,
- it does not recommend new skills,
- it only measures support for the skills already present in the `skills` list.

That keeps the system aligned to the project brief and easier to defend methodologically.

## Step 10: Convert cosine similarity into a competency score
Cosine similarity is used as the core evidence-alignment measure. It is then converted into:

- `similarity_score`
- `competency_score`
- `score_band`

The `competency_score` is currently a simple scaled version of similarity, and the `score_band` is assigned from configured thresholds. This was chosen because it is transparent, reproducible, and easy to explain in an MVP.

## Step 11: Keep a full audit trail for explainability
Each output row includes:

- `talentlinkId`
- `skill`
- `cleaned_description`
- `reference_text`
- `similarity_score`
- `competency_score`
- `score_band`
- `evidence_excerpt`

This was done to make the output defensible. A reviewer can trace each score back to:

- the source employee,
- the listed skill,
- the cleaned text used,
- the benchmark text used,
- the extracted evidence sentence.

## Step 12: Test the critical edges, not just the happy path
The tests were designed around the points most likely to undermine the methodology if they failed:

- missing required columns,
- malformed skill values,
- noisy text normalization,
- null handling,
- row aggregation,
- score generation.

This matters for the final project because it shows the implementation was tested systematically instead of only being demonstrated once in a notebook.

## Summary
The implementation was built by following the project specification strictly and choosing the simplest defensible design at each stage:

1. normalize the input schema,
2. validate early,
3. aggregate to employee level,
4. clean text conservatively,
5. compare listed skills against ideal references,
6. use TF-IDF plus cosine similarity for transparent scoring,
7. preserve a full audit trail,
8. verify the critical logic with tests.

That sequence is what makes the current MVP both explainable and methodologically disciplined.
