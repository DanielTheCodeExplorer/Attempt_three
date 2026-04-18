# Competency Scoring MVP: Module Guide

## Purpose of This Note
This document explains the current module structure for the competency scoring pipeline. It reflects the latest implementation, where the system builds one final competency dataset by combining:

- biography TF-IDF evidence,
- job-history TF-IDF evidence,
- LLM-derived confidence by source,
- summed competency scores and rank per employee-skill pair.

## Why the code was modularised
The solution was implemented as a modular pipeline rather than one long script to improve:

- clarity,
- reproducibility,
- testability,
- maintainability,
- explainability in the final report.

Each module has one main responsibility.

## `src/config.py`
This file stores central configuration for the pipeline.

It contains:

- input and output paths,
- required field names,
- input column aliases,
- OpenAI model settings,
- score thresholds,
- taxonomy and skill-profile settings used by TF-IDF scoring.

Why this module exists:

- to avoid hardcoding settings across multiple files,
- to make experiments easier to reproduce,
- to keep scoring assumptions visible in one place.

## `src/logging_utils.py`
This file configures structured logging for the pipeline.

It provides:

- a reusable logger getter,
- a common logging format,
- one logging configuration entrypoint.

Why this module exists:

- to replace ad hoc `print` statements,
- to leave an execution trail for loading, normalization, extraction, and output generation,
- to make failures easier to diagnose.

## `src/data_loader.py`
This file handles data ingestion and early validation.

Its responsibilities are:

- reading the main CSV file,
- renaming known input aliases to canonical names,
- checking that `talentlinkId`, `description`, and `skills` exist,
- converting descriptions into safe text values,
- parsing the `skills` field into a list,
- removing unusable rows,
- aggregating repeated rows to one employee-level record.

Why this module exists:

- validation needs to happen before scoring,
- input cleanup should be separate from scoring logic,
- one employee-level dataset is easier to score consistently.

## `src/biography_dataset.py`
This file builds the biography-specific dataset used by the biography pipeline and the final combined dataset.

Its responsibilities are:

- grouping biography fields to employee level,
- preserving `talentlinkId`, `biography`, and `skills`,
- returning a biography-focused dataframe without forcing intermediate CSV output.

Why this module exists:

- biography processing is now a separate evidence source,
- it is cleaner to isolate biography assembly from the main job-history loader.

## `src/text_preprocessing.py`
This file applies lightweight text normalization.

Its responsibilities are:

- lowercasing,
- punctuation cleanup,
- whitespace normalization,
- skill-name normalization.

Why this module exists:

- text cleaning rules should be explicit and reusable,
- the scorer should not be cluttered with string manipulation logic,
- normalization improves consistency before both LLM extraction and TF-IDF scoring.

## `src/skill_taxonomy.py`
This file defines the canonical skill taxonomy.

It provides:

- the canonical skill list,
- alias definitions,
- skill relationships,
- CSV export helpers for taxonomy artifacts.

Why this module exists:

- the scorer needs one controlled skill inventory,
- aliases and relationships should be explicit and reusable,
- taxonomy artifacts support explainability in the project report.

## `src/skill_extraction.py`
This file handles taxonomy-based skill matching.

It provides:

- text preprocessing for extraction,
- alias and phrase matching,
- evidence snippet collection,
- optional embedding fallback when alias matches are missing.

Why this module exists:

- skill detection should be separated from scoring,
- the pipeline needs a controlled fallback for unseen wording,
- extraction evidence is useful for auditing and diagnostics.

## `src/llm_normalizer.py`
This file contains the LLM-based evidence and confidence layer.

It provides:

- the normalization and extraction prompts,
- `normalize_biography_text(...)`,
- `normalize_biography_dataframe(...)`,
- `extract_supported_skills_from_text(...)`,
- structured validation of the LLM JSON response,
- conservative fallback behavior when no LLM client is available.

Why this module exists:

- the final dataset still uses the LLM to judge evidence strength,
- confidence labels must remain controlled and auditable,
- the fallback path must preserve pipeline stability when the API is unavailable.

## `src/competency_scoring.py`
This file contains TF-IDF-based skill scoring utilities.

It provides:

- skill profile preparation,
- TF-IDF vectorization,
- cosine-style similarity scoring per skill,
- taxonomy-aware scoring helpers used by downstream builders.

Why this module exists:

- TF-IDF scoring should remain reusable across job-history and biography evidence,
- text-similarity logic should stay separate from final dataset assembly,
- the project still needs per-skill score generation in a controlled module.

## `src/final_competency_dataset.py`
This is the core final-dataset builder.

It contains the logic that:

- loads job-history and biography evidence,
- aligns both sources to the canonical skill list,
- computes biography TF-IDF scores,
- computes job-history TF-IDF scores,
- derives biography confidence from LLM evidence strength,
- derives job-history confidence from LLM evidence strength,
- calculates source-level competency scores,
- sums the competency scores,
- ranks skills within each employee.

Why this module exists:

- the final dissertation dataset needs one stable builder,
- the final scoring formula should live in one auditable place,
- ranking and summary logic should not be duplicated across scripts.

## `src/run_final_competency_dataset.py`
This file is the entrypoint for building the final consolidated dataset.

It performs:

1. configure logging,
2. build the final dataset dataframe,
3. write `data/outputs/final_competency_dataset.csv`.

Why this module exists:

- the final dataset should be reproducible with one command,
- orchestration should stay separate from implementation details.

## `src/main.py`
This file remains the main job-history scoring entrypoint.

It performs:

1. configure logging,
2. load and prepare the main dataset,
3. score the job-history evidence,
4. write the main score output.

Why this module exists:

- the project still needs a standalone job-history run,
- the earlier output remains useful for source-level comparison.

## `src/run_biography_pipeline.py`
This file is the biography pipeline entrypoint.

It performs:

1. biography dataset creation,
2. biography normalization,
3. writing the normalized biography dataset,
4. biography-side scoring,
5. writing the biography score output.

Why this module exists:

- biography remains a distinct evidence source,
- the final dataset depends on this source-specific processing.

## `tests/test_competency_scoring.py`
This file contains the main unit tests for TF-IDF skill scoring.

It checks:

- skill scoring behavior,
- source-specific scoring outputs,
- expected per-skill score calculations.

## `tests/test_final_competency_dataset.py`
This file contains the main unit tests for the consolidated final dataset.

It checks:

- expected output columns,
- confidence score calculation,
- summed TF-IDF score calculation,
- source-specific competency score calculation,
- summed competency score calculation,
- rank assignment,
- `-1` ranking for zero-score rows.

Why these test modules exist:

- to show the pipeline was tested in a disciplined way,
- to guard the contract of the core modules,
- to support reproducibility and confidence in the implementation.

## How the modules work together
The current final-dataset workflow runs in this order:

1. `run_final_competency_dataset.py` starts the final dataset build.
2. `config.py` supplies paths, model settings, and scoring assumptions.
3. `data_loader.py` prepares employee-level job-history input.
4. `biography_dataset.py` prepares employee-level biography input.
5. `text_preprocessing.py` standardizes text and skill labels.
6. `llm_normalizer.py` derives confidence-relevant evidence strength from each source.
7. `competency_scoring.py` computes biography and job-history TF-IDF scores against the canonical skill profiles.
8. `final_competency_dataset.py` combines both evidence sources into one final row per employee-skill pair.
9. the resulting CSV acts as the final audit dataset for downstream analysis.
10. `tests/test_final_competency_dataset.py` verifies that the final dataset contract behaves as expected.

## Why this structure supports the methodology section
This structure makes the system easier to explain in the project report because it separates:

- data validation,
- biography preparation,
- job-history preparation,
- LLM confidence assignment,
- TF-IDF similarity scoring,
- source-level competency calculation,
- summed competency calculation,
- employee-level skill ranking,
- testing.

That separation is useful academically as well as technically. It allows the report to explain each stage clearly and justify why the workflow was designed in that order.
