# Skill-Specific Profile Generator

## Purpose
This stage converts a broad employee description into a target-skill profile using a retrieve-then-rewrite flow.

The design goal is to improve focus without inventing evidence. Instead of rewriting from the whole profile, the pipeline first retrieves the chunks most relevant to the target skill and only then produces a narrowed profile.

## Input Contract
Each row is expected to contain:

- `talentlinkId`
- `description`
- `skills`
- `target_skill`

The `description` field is treated as the only source evidence text.

## Core Steps
The implementation lives in `src/skill_specific_profile.py` and follows this sequence:

1. `load_data()`
2. `preprocess_text()` from `src/text_preprocessing.py`
3. `chunk_profile_text()`
4. `rank_chunks_for_skill()`
5. `generate_skill_specific_profile()`
6. `build_output_record()`

## Retrieval Logic
The retrieval step is deterministic and explainable.

For each chunk, the scorer builds a relevance score from:

- direct match to the target skill,
- configured aliases for that skill,
- token overlap with the skill reference text,
- token overlap with the target-skill vocabulary.

Only the top chunks above a minimum relevance threshold are kept. This prevents irrelevant parts of the profile from shaping the rewritten output.

## Rewrite Logic
The rewrite step operates only on the retrieved evidence spans.

If no safe LLM client is available, the system falls back to an evidence-only profile built from the retrieved chunks. If an LLM is available, it receives:

- the target skill,
- the configured reference text for that skill,
- the retrieved evidence spans only.

This keeps the rewrite scope narrow and reduces the chance of broad profile embellishment.

## Hallucination Controls
Hallucination risk is reduced in three ways:

- retrieval limits the prompt to relevant evidence instead of the full profile,
- the prompt explicitly forbids adding projects, credentials, outcomes, or expertise not present in the evidence,
- generated output is sanitized against the retrieved evidence and falls back to the evidence-only version if unsupported content appears.

## Output Contract
Each output row contains:

- `talentlinkId`
- `target_skill`
- `skill_specific_profile`
- `evidence_spans`
- `confidence_score`

This keeps the result auditable. A reviewer can inspect which original spans were retrieved and how strongly the system judged them to support the target skill.
