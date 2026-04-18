# Competency Scoring MVP: Step-by-Step Reasoning

## Purpose of This Note
This document explains the current reasoning behind the final competency scoring implementation. It reflects the updated workflow, where the system uses:

- biography TF-IDF scoring,
- job-history TF-IDF scoring,
- LLM-derived confidence labels,
- summed competency scores,
- per-employee skill ranking.

## Step 1: Start from the revised project constraint
The earlier workflow changed several times because a single evidence source was not enough:

- job-history text was often too generic,
- biography text was richer but still broad,
- extraction-only scoring removed TF-IDF from the active workflow,
- the project still needed a final interpretable dataset that preserved textual similarity and evidence confidence together.

The revised final design therefore combines both:

- TF-IDF similarity as the text-alignment signal,
- LLM confidence as the evidence-strength signal.

## Step 2: Keep the input schema stable
The final pipeline still starts from the same main employee data contract:

- `talentlinkId`
- `description`
- `skills`

It also adds biography data as a second evidence source during dataset assembly.

This keeps the rest of the project stable even though the internal scoring method changed again.

## Step 3: Validate the schema before scoring
Validation still happens first because every later stage depends on the input fields being usable.

The loader checks:

- whether the required columns exist,
- whether descriptions can be converted safely to text,
- whether the `skills` field can be converted into a proper list,
- whether empty rows should be excluded.

This keeps failures early and predictable.

## Step 4: Collapse repeated rows to one employee record
The source dataset is row-based, but the final scoring problem is employee-level.

For that reason, the loader aggregates repeated rows by `talentlinkId` and builds:

- one combined job-history description per employee,
- one deduplicated list of declared skills per employee.

The biography pipeline performs the same employee-level grouping for biographies.

This means the scorer evaluates each employee's available evidence as a whole, rather than treating each raw row as an independent case.

## Step 5: Parse the declared skills into a controlled list
The `skills` field may arrive in different forms:

- already as a Python list,
- as a stringified list,
- as comma-separated text.

The parser standardizes these cases, deduplicates repeated skills, and preserves order. This keeps the scoring contract stable and gives the final dataset a reliable declared-skill reference.

## Step 6: Use conservative text preprocessing
The preprocessing stage stays intentionally simple:

- lowercase text,
- remove punctuation noise,
- normalize whitespace,
- normalize skill labels.

The goal is not to transform meaning. The goal is to make both LLM extraction and TF-IDF scoring more consistent.

## Step 7: Score biography and job history separately with TF-IDF
The final design uses two source-specific TF-IDF similarity signals:

- `biography_tfidf_score`
- `job_description_tfidf_score`

Each score measures how closely the available source text aligns with the canonical skill profile for that skill.

This preserves the original methodological idea that skill evidence can be compared against a predefined skill profile using TF-IDF-based similarity.

## Step 8: Use the LLM to assign source-level evidence confidence
The LLM is still used, but its role is narrower than in earlier versions.

It is used to identify supported skills and assign an evidence-strength label for each source:

- `high`
- `medium`
- `low`

These are then mapped to numeric confidence values:

- `high -> 1.0`
- `medium -> 0.7`
- `low -> 0.4`

This means the LLM is not directly generating the competency score. It is supplying one part of the final formula: confidence.

## Step 9: Calculate source-level competency scores
The system calculates a competency score separately for each source.

The formulas are:

- `biography_competency_score = biography_tfidf_score * biography_confidence_score * 100`
- `job_description_competency_score = job_description_tfidf_score * job_description_confidence_score * 100`

This design preserves two ideas at once:

- TF-IDF measures textual alignment,
- LLM confidence measures strength of evidence.

Multiplying them ensures that a high textual match with weak evidence is reduced, and that stronger evidence has more influence.

## Step 10: Calculate shared helper columns
The final dataset also includes summary helper values:

- `confidence_score = ((biography_confidence_score + job_description_confidence_score) / 2) * 100`
- `sum_tfidf_score = biography_tfidf_score + job_description_tfidf_score`
- `sum_competency_score = biography_competency_score + job_description_competency_score`

These make the final CSV easier to interpret because they show:

- average confidence across the two sources,
- total TF-IDF evidence across the two sources,
- total competency contribution across the two sources.

## Step 11: Rank skills within each employee
The final dataset ranks each skill within an employee using:

- `competency_score_rank`

This rank is based on descending `sum_competency_score`.

If `sum_competency_score == 0`, then:

- `competency_score_rank = -1`

This makes the output easier to interpret because the rank now answers:

- which skills appear strongest for this employee across both evidence sources?

## Step 12: Keep a final consolidated audit trail
Each output row now includes:

- `talentlinkId`
- `skill_id`
- `skill`
- `category`
- `declared_skills_json`
- `is_declared_skill`
- `position_text`
- `job_history_text`
- `biography_text`
- `combined_text`
- `biography_tfidf_score`
- `job_description_tfidf_score`
- `biography_confidence_score`
- `job_description_confidence_score`
- `confidence_score`
- `sum_tfidf_score`
- `biography_competency_score`
- `job_description_competency_score`
- `sum_competency_score`
- `competency_score_rank`

This makes the output easier to defend because each score can be traced back to:

- the employee,
- the skill,
- the declared skill list,
- the available evidence text,
- the source-specific TF-IDF values,
- the source-specific confidence values,
- the final summed competency value and rank.

## Step 13: Test the contract, not just the happy path
The tests focus on the areas most likely to undermine the methodology if they fail:

- missing required columns,
- malformed skill values,
- noisy text normalization,
- row aggregation,
- source-level TF-IDF calculation,
- source-level confidence mapping,
- competency score calculation,
- summed competency score calculation,
- competency rank assignment,
- `-1` rank assignment for zero-score rows.

This demonstrates that the workflow is systematic rather than a one-off notebook experiment.

## Summary
The current implementation follows this sequence:

1. normalize the input schema,
2. validate early,
3. aggregate to employee level,
4. clean text conservatively,
5. score biography text with TF-IDF,
6. score job-history text with TF-IDF,
7. derive LLM-based confidence per source,
8. calculate source-level competency scores,
9. sum the competency scores,
10. rank skills within each employee,
11. preserve a full final audit trail,
12. verify the contract with tests.

This final workflow is more aligned with the project’s combined methodology than the earlier extraction-only scoring stage because it brings TF-IDF back into the active score while still retaining LLM-based evidence strength.
