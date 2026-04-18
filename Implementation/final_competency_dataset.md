# Final Competency Dataset

## Purpose
This note explains the structure and logic of the final consolidated dataset used for competency analysis.

The dataset combines:

- job-history evidence,
- biography evidence,
- TF-IDF similarity scoring,
- LLM-derived confidence scoring,
- per-employee skill ranking.

The output file is:

- `data/outputs/final_competency_dataset.csv`

## Row Structure
The dataset contains one row per:

- `talentlinkId + skill`

This means each employee has one row for every canonical skill in the taxonomy.

## Core Columns
The final dataset currently contains:

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

## Meaning of the Score Columns
### `biography_tfidf_score`
This is the TF-IDF similarity score between the employee's biography text and the canonical skill profile for that skill.

### `job_description_tfidf_score`
This is the TF-IDF similarity score between the employee's job-history text and the canonical skill profile for that skill.

### `biography_confidence_score`
This is the numeric confidence score derived from LLM evidence strength for the biography source.

### `job_description_confidence_score`
This is the numeric confidence score derived from LLM evidence strength for the job-history source.

### Confidence mapping
The evidence-strength mapping is:

- `high -> 1.0`
- `medium -> 0.7`
- `low -> 0.4`

## Formulae
### Average confidence helper
```text
confidence_score =
((biography_confidence_score + job_description_confidence_score) / 2) * 100
```

### Summed TF-IDF helper
```text
sum_tfidf_score =
biography_tfidf_score + job_description_tfidf_score
```

### Biography competency
```text
biography_competency_score =
biography_tfidf_score * biography_confidence_score * 100
```

### Job-history competency
```text
job_description_competency_score =
job_description_tfidf_score * job_description_confidence_score * 100
```

### Summed competency
```text
sum_competency_score =
biography_competency_score + job_description_competency_score
```

## Ranking Logic
The dataset ranks skills within each employee using:

- `competency_score_rank`

This rank is based on descending `sum_competency_score`.

If:

- `sum_competency_score == 0`

then:

- `competency_score_rank = -1`

This indicates that no evidence was found for that skill across either source.

## Why this dataset is useful
This dataset is designed to support competency analysis in a way that is:

- explainable,
- source-aware,
- auditable,
- compatible with later statistical or modelling work.

It makes it possible to inspect:

- whether the biography supports a skill,
- whether the job history supports a skill,
- whether both sources support the same skill,
- which skills rank highest for each employee.

## Interpretation Example
If a row has:

- `biography_tfidf_score = 0.08`
- `job_description_tfidf_score = 0.03`
- `biography_confidence_score = 0.7`
- `job_description_confidence_score = 0.4`

then:

- `biography_competency_score = 0.08 * 0.7 * 100 = 5.6`
- `job_description_competency_score = 0.03 * 0.4 * 100 = 1.2`
- `sum_competency_score = 6.8`
- `confidence_score = ((0.7 + 0.4) / 2) * 100 = 55`

This means the skill is supported more strongly by the biography than the job history, and the final rank will depend on how `6.8` compares with the employee's other skill rows.
