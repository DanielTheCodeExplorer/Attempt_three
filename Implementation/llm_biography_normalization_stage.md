# LLM Biography Normalization Stage

## Purpose
This note explains the new LLM-based normalization stage added before competency scoring in the biography pipeline.

The aim of this stage is to take raw biography text and rewrite it into a more standardised skill-evidence format so the downstream scoring model receives cleaner and more comparable input.

## New Module
The new module is:

- `src/llm_normalizer.py`

It provides:

- the exact LLM extraction prompt,
- `normalize_biography_text(...)`,
- `normalize_biography_dataframe(...)`,
- structured validation of the LLM JSON response,
- conservative fallback behavior if the LLM call is unavailable or fails.

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

## Exact Prompt Used
The exact prompt is stored in `src/llm_normalizer.py` as `LLM_NORMALIZER_PROMPT`.

It instructs the model to:

- use only explicit information from the biography,
- avoid inventing skills,
- restrict skills to the allowed list when provided,
- prefer precision over recall,
- return valid JSON only.

## Integration Point
The integration point is the biography pipeline in:

- `src/run_biography_pipeline.py`

The updated sequence is now:

1. create the base biography dataset,
2. run LLM normalization over each biography,
3. save a normalized biography dataset,
4. score using `clean_skill_evidence_text` through the normal pipeline,
5. write biography-based score and evaluation outputs.

## How the scorer uses normalized text
The loader already expects a `description` field. The normalization stage now creates:

- `clean_skill_evidence_text`
- `description`

The `description` field is set from `clean_skill_evidence_text` when available, so the scorer uses normalized biography evidence instead of the raw biography text.

## Configuration
The model name is configured through the environment variable:

- `OPENAI_MODEL_NAME`

This is read into `PipelineConfig.llm_model_name`.

The OpenAI API key is expected through:

- `OPENAI_API_KEY`

## Error Handling
The normalizer includes:

- logging,
- JSON validation,
- filtering of any skills not in the allowed list,
- conservative fallback behavior when the LLM is unavailable or an exception occurs.

The fallback is intentionally strict:

- it returns fewer skills rather than more,
- it only keeps explicit direct skill matches found in the source biography.

## Files Changed
- `src/config.py`
- `src/llm_normalizer.py`
- `src/run_biography_pipeline.py`
- `requirements.txt`
- `pyproject.toml`
- `tests/test_llm_normalizer.py`

## Example Input
Raw biography:

`Built Python automation for reporting and queried data using SQL.`

Allowed skills:

- `Python`
- `SQL`

## Example Output
```json
{
  "standardized_summary": "Delivers Python automation and SQL reporting work.",
  "matched_skills": [
    {
      "skill": "Python",
      "evidence_phrase": "Python automation",
      "evidence_sentence": "Built Python automation for reporting.",
      "evidence_strength": "high"
    },
    {
      "skill": "SQL",
      "evidence_phrase": "SQL reporting",
      "evidence_sentence": "Queried data using SQL for reporting.",
      "evidence_strength": "medium"
    }
  ],
  "clean_skill_evidence_text": "Python automation for reporting. SQL reporting and data querying."
}
```

## Why this was added
The earlier scoring pipeline used raw biography or description text directly. That created inconsistent wording and made lexical matching brittle.

The LLM normalization stage improves this by:

- rewriting biographies into standardised evidence wording,
- extracting only supported skill evidence,
- creating a cleaner text field for downstream similarity scoring,
- preserving explainability through structured matched-skill evidence.
