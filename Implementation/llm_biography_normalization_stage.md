# LLM Biography Normalization Stage

## Purpose
This note explains the current LLM-based normalization stage used in the biography pipeline and the final competency dataset.

The purpose of this stage is to convert biography text into structured, auditable skill evidence before downstream TF-IDF scoring and confidence assignment. In the current workflow, the LLM is not the final scorer. Instead, it provides structured skill evidence and evidence-strength labels that feed into the final competency calculations.

## Main Module
The main module is:

- `src/llm_normalizer.py`

It provides:

- the extraction prompt,
- `normalize_biography_text(...)`,
- `normalize_biography_dataframe(...)`,
- `extract_supported_skills_from_text(...)`,
- structured validation of the LLM JSON response,
- conservative fallback behavior when the LLM client is unavailable.

## Required JSON Output
The normalizer returns JSON with:

- `standardized_summary`
- `matched_skills`
- `clean_skill_evidence_text`

Each `matched_skills` entry contains:

- `skill`
- `evidence_phrase`
- `evidence_sentence`
- `evidence_strength`

## Prompt Behavior
The exact prompt is stored in `src/llm_normalizer.py` as `LLM_NORMALIZER_PROMPT`.

It instructs the model to:

- use only explicit information from the source text,
- avoid inventing skills,
- restrict skills to the allowed list when provided,
- prefer precision over recall,
- return valid JSON only,
- return fewer skills rather than guessing if evidence is weak.

## Integration Points
The main integration points are:

- `src/run_biography_pipeline.py`
- `src/final_competency_dataset.py`

The current biography sequence is:

1. create the base biography dataset,
2. run normalization over each employee biography,
3. save the normalized biography dataset,
4. keep the biography evidence text available for downstream scoring,
5. derive biography-side evidence strength from the matched skills,
6. combine this with biography TF-IDF scoring in the final dataset.

## How the final scorer uses normalized biography text
The biography normalizer creates:

- `standardized_summary`
- `matched_skills_json`
- `clean_skill_evidence_text`

The normalized dataframe then sets:

- `description = clean_skill_evidence_text` when available

This means downstream scoring evaluates biography-derived evidence text rather than the raw biography block wherever possible.

In the current final workflow, the biography evidence is used in two ways:

- TF-IDF scoring against the canonical skill profiles,
- LLM-derived evidence confidence via `evidence_strength`.

The LLM therefore supports the final score, but does not replace the TF-IDF component.

## Why this changed
The earlier design attempted to use the LLM as a stronger extraction-first scorer. In practice, the project still needed a final dataset that combined:

- textual similarity,
- evidence strength,
- source-by-source comparison.

For that reason, the role of the biography normalizer became narrower and more controlled. It now focuses on:

- structuring biography evidence,
- preserving auditability,
- assigning evidence strength,
- supporting downstream TF-IDF-based competency calculations.

## Confidence Mapping
The final dataset maps `evidence_strength` to numeric confidence values:

- `high -> 1.0`
- `medium -> 0.7`
- `low -> 0.4`

These values are then used for:

- `biography_confidence_score`
- and, through the same extraction logic, `job_description_confidence_score`

The final shared helper column is:

- `confidence_score = ((biography_confidence_score + job_description_confidence_score) / 2) * 100`

## Configuration
The model name is configured through:

- `OPENAI_MODEL_NAME`

The optional base URL is configured through:

- `OPENAI_BASE_URL`

The API key is expected through:

- `OPENAI_API_KEY`

If no working client is available, the module uses its conservative fallback path.

## Error Handling
The normalizer includes:

- logging,
- JSON validation,
- filtering of any skills not in the allowed list,
- conservative fallback behavior when the LLM is unavailable or an exception occurs.

The fallback is intentionally strict:

- it returns fewer skills rather than more,
- it avoids unsupported matches,
- it keeps the pipeline operational in offline environments.

## Files Changed
- `src/config.py`
- `src/llm_normalizer.py`
- `src/run_biography_pipeline.py`
- `src/final_competency_dataset.py`
- `tests/test_llm_normalizer.py`
- `tests/test_final_competency_dataset.py`

## Example Input
Biography:

`Built Python automation for reporting and queried data using SQL for reconciliations.`

Allowed skills:

- `Python`
- `SQL`
- `Power BI`

## Example Output
```json
{
  "standardized_summary": "Built reporting automation and data-querying support work.",
  "matched_skills": [
    {
      "skill": "Python",
      "evidence_phrase": "Python automation",
      "evidence_sentence": "Built Python automation for reporting and queried data using SQL for reconciliations.",
      "evidence_strength": "high"
    },
    {
      "skill": "SQL",
      "evidence_phrase": "queried data using SQL",
      "evidence_sentence": "Built Python automation for reporting and queried data using SQL for reconciliations.",
      "evidence_strength": "medium"
    }
  ],
  "clean_skill_evidence_text": "Python automation for reporting. Queried data using SQL for reconciliations."
}
```

## Why this stage matters
This stage improves the final pipeline by:

- turning broad biography text into structured evidence,
- restricting output to skills supported by the source text,
- creating cleaner biography evidence for downstream scoring,
- preserving explainability through phrase-level and sentence-level evidence,
- supplying confidence values that are later combined with TF-IDF scores in the final competency dataset.
