# Competency Scoring MVP: Module Guide

## Purpose of This Note
This document explains what each module in the competency scoring MVP does and why the implementation was split this way. The aim is to support the methodology and software engineering discussion in the project write-up.

## Why the code was modularised
The solution was deliberately implemented as a modular pipeline rather than as one long script. This structure was chosen to improve:

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
- TF-IDF settings,
- score bands,
- known skill reference descriptions,
- a fallback reference template.

Why this module exists:

- to avoid hardcoding settings across multiple files,
- to make experiments easier to reproduce,
- to keep model assumptions visible in one place.

## `src/logging_utils.py`
This file configures structured logging for the pipeline.

It provides:

- a reusable logger getter,
- a common logging format,
- one logging configuration entrypoint.

Why this module exists:

- to replace ad hoc `print` statements,
- to leave an execution trail for loading, scoring, and output generation,
- to make failures easier to diagnose.

## `src/data_loader.py`
This file handles data ingestion and early validation.

Its responsibilities are:

- reading the CSV file,
- renaming known input aliases to canonical names,
- checking that `talentlinkId`, `description`, and `skills` exist,
- converting descriptions into safe text values,
- parsing the `skills` field into a list,
- removing unusable rows,
- aggregating repeated rows to one employee-level record.

Why this module exists:

- validation needs to happen before scoring,
- input cleanup should be separate from the scoring logic,
- one employee-level dataset is easier to score consistently.

## `src/text_preprocessing.py`
This file applies lightweight text normalization and evidence extraction.

Its responsibilities are:

- lowercasing,
- punctuation cleanup,
- whitespace normalization,
- skill-name normalization,
- selecting an evidence sentence for the audit trail.

Why this module exists:

- text cleaning rules should be explicit and reusable,
- the scorer should not be cluttered with string manipulation logic,
- evidence extraction is part of explainability rather than vectorization.

## `src/skill_references.py`
This file resolves the benchmark description used for each skill.

Its job is:

- to look up a configured ideal reference description,
- or generate a fallback reference if no explicit entry exists.

Why this module exists:

- the AES-style design depends on comparing employee text to an ideal answer,
- reference handling should be isolated so it can be expanded later without changing the scorer class.

## `src/competency_scoring.py`
This is the core scoring module.

It contains:

- the `ScoreRecord` dataclass,
- the `CompetencyScorer` class,
- TF-IDF vectorization,
- cosine similarity scoring,
- score conversion and banding,
- audit trail row generation.

Why this module exists:

- the project needed a reusable class-based design,
- the central scoring logic should live in one controlled place,
- the scorer needs to remain separate from data loading and configuration.

## `src/evaluation.py`
This file provides a simple analytical summary over the scoring output.

It currently groups results by skill and reports:

- average similarity,
- average competency score,
- number of high-band results.

Why this module exists:

- raw scoring rows are useful operationally,
- summarized results are useful analytically,
- keeping summary logic separate prevents the scorer from taking on multiple roles.

## `src/main.py`
This file is the pipeline entrypoint.

It performs the top-level execution order:

1. configure logging,
2. load and prepare the dataset,
3. create a `CompetencyScorer`,
4. score the employee-skill pairs,
5. write the results to CSV.

Why this module exists:

- the full workflow needs a clean runnable entrypoint,
- orchestration should be separated from implementation details inside the modules.

## `tests/test_competency_scoring.py`
This file contains the MVP unit tests.

It checks:

- schema validation,
- skill parsing,
- preprocessing behavior,
- input aggregation,
- generation of audit-trail scoring output.

Why this module exists:

- to show the pipeline was tested in a disciplined way,
- to guard the contract of the core modules,
- to support reproducibility and confidence in the implementation.

## How the modules work together
The current pipeline runs in this order:

1. `main.py` starts the pipeline.
2. `config.py` supplies the rules and settings.
3. `data_loader.py` validates and prepares the employee-level input.
4. `text_preprocessing.py` cleans descriptions and supports evidence extraction.
5. `skill_references.py` supplies the benchmark text for each skill.
6. `competency_scoring.py` vectorizes, compares, and scores.
7. `evaluation.py` can summarize the results for analysis.
8. `tests/test_competency_scoring.py` verifies that the core logic behaves as expected.

## Why this structure supports the methodology section
This modular structure makes the system easy to explain in a final-year project because it separates:

- data validation,
- text transformation,
- benchmark reference generation,
- numerical comparison,
- score generation,
- result auditing,
- testing.

That separation is useful academically as well as technically. It allows the report to explain each stage clearly and justify why the pipeline was designed in that order.
