# Generating Dummy Data - Implementation Review

**Implementation Date: March 12, 2026**

This section reviews the implementation of the project and demonstrates practical mastery of data engineering, Python programming, and information system quality controls.

## Project Objective
The objective was to build a synthetic Excel workbook generator that reproduces the structure and statistical behaviour of the source schema without copying real personal data. The implementation was completed in a single production-style script: `generate_synthetic_workbook.py`.

## Core Implementation Work
The implementation included the following material technical outputs:
- Parsing schema metadata (`sheet_name`, header row index, column order, null ratios, gap runs, length profiles, repetition profiles).
- Generating a person-centric synthetic population with consistent identities across repeated rows.
- Enforcing derived relationships such as email-from-name and prefixed IDs (`r` + employee ID).
- Reproducing one-to-many record expansion for skills, certifications, job history, education, and travel preferences.
- Building clustered missing-data masks from `gap_profile` rather than using random null placement.
- Applying logical date constraints (certification expiry after issue date; job end date after job start date when present).
- Writing output to Excel with the header row in the same position as the profiled workbook.

## Key Challenges and How They Were Solved
1. Challenge: Preserving realistic repeated-record behaviour per employee.
Solution: A base employee table was generated first, then expanded into multiple rows per employee using controlled row-block assignment. This kept IDs, names, and emails consistent across all repeated records.

2. Challenge: Matching missing data patterns that were clustered (leading, internal, trailing runs).
Solution: Per-column masks were built from schema `gap_runs` and then adjusted to target `null_count`. This preserved both the shape of missing blocks and overall sparsity ratios.

3. Challenge: Maintaining data integrity across dependent date fields.
Solution: Dependency checks were added so end/expiration values are only populated where their corresponding start/issue context exists; generated dates were constrained to valid chronological order.

4. Challenge: Producing varied text lengths without copying source values.
Solution: Text generation used metadata-driven target lengths and mode-based generation (short phrases, medium sentences, long paragraphs), then trimming/padding to align with column profiles.

5. Challenge: Reproducing categorical frequency behaviour.
Solution: Weighted synthetic category pools were used to emulate high-repetition, near-constant, and higher-variance columns while avoiding literal value reuse.

## Knowledge and Skills Demonstrated
This implementation demonstrates:
- Practical Python engineering with `pandas`, `numpy`, `random`, `datetime`, and `Faker`.
- Data modelling for entity consistency and relationship integrity.
- Statistical pattern replication from schema metadata.
- Data quality controls for null behaviour, typing, and temporal logic.
- Workbook-oriented data delivery for business reporting environments.

## Evidence of Professional Technical Practice
Evidence produced in this implementation includes:
- Source code artifact: `generate_synthetic_workbook.py`.
- Documentation artifact: `Implementation/generating_dummy_data_implementation_review.md`.
- Data process outcome: synthetic workbook generation matching schema-defined structure and behaviour.

These outputs provide direct evidence of material work with code, data, and information-system constraints in line with apprenticeship expectations.

## Why This Approach Was Chosen
A metadata-driven generator was selected over a manually scripted table because it is:
- More robust when schema details change.
- More compliant with privacy constraints (synthetic values only).
- More defensible in technical review because behaviour is tied directly to profiled metadata.

## Update Log
**Update Date: March 12, 2026**

The `Summary/bio` generation was improved to address low-quality narrative output. The script now:
- Supports optional style inference from external example bios (via Polars, using the `Moreinfo` field when available).
- Uses a structured professional-bio generator with realistic consulting language, strengths, delivery impact, and tooling context.
- Keeps outputs fully synthetic while improving readability and professional tone.

**Update Date: March 12, 2026 (Performance Refactor)**

The generator was refactored to reduce runtime while preserving schema-driven output behaviour:
- Replaced row-by-row `pandas.DataFrame.at` writes with column-backed Python lists, then constructed the DataFrame once at the end.
- Kept the same mask logic, dependency rules, and output schema, but applied null masks using array operations before DataFrame creation.
- Moved repeated constant/category setup outside inner loops (for example skill level/proficiency pools) to reduce per-row overhead.

This change improved execution efficiency without changing the intended workbook structure, relationships, or sparsity constraints.

**Update Date: March 13, 2026 (Skill Level 1 Rule Correction)**

The generation logic was updated to enforce revised domain rules for `Skill Level 1` and related columns:
- `Skill Level 1` now uses only five categories: `skills`, `language`, `certification/credentials`, `job history`, and `education`.
- `Capability Name` is explicitly left blank for `skills` rows.
- Language handling was made person-consistent:
  - all people have English,
  - a small subset has one extra language,
  - two people can have up to five languages.
- Proficiency rules were corrected for language rows, including English bias toward `Upper Advanced` with a small `Advanced` minority.
- Education columns are now conditionally populated only when `Skill Level 1 = education`:
  - `Education title/type of degree`,
  - `Area of Study`,
  - `Academic Institution`,
  - `In Process`.
- `Academic Institution` is always populated for education rows and selected from real institution names.
- `In Process` is constrained to three people and only for Bachelor/Master pathways.
- Job history columns are filled only on `job history` rows with one-to-many per-person behavior and capped ranges.
- Travel logic was changed so travel appears at most once per person, with approximately 80% of populated travel rows set to `Yes`.

Technical note:
- A major performance bottleneck was removed by replacing a huge `numpy.arange` ID allocation with efficient range sampling, reducing runtime to seconds while preserving output structure.

**Update Date: March 16, 2026 (Strict Conditional Rebuild for Skill Level 1)**

The workbook generator was rebuilt to enforce strict conditional logic tied to `Skill Level 1` and to output a fresh corrected dataset (`synthetic_staff_talentlink_v3.xlsx`).

Key implementation changes:
- Enforced exact `Skill Level 1` labels:
  - `Skills`
  - `Language`
  - `Certifications/Credentials`
  - `Job History`
  - `Education`
  - `Travel`
- Implemented largest-remainder distribution targeting from requested percentages, producing exact row counts per type.
- Added row-type allocation with person-level constraints so one-to-many behavior remains realistic.
- Added strict post-generation blanking by row type so non-applicable controlled fields are forcibly set to blank.
- Enforced date validity and ordering:
  - Certification expiration never before issued.
  - Job end date never before start date.
- Enforced field requirements:
  - Job History start/position/company always populated.
  - Education academic institution and in-process always populated.
- Maintained one-travel-row-per-person cap and `Travel Interest` value rule (`Yes` only when populated).

Validation summary on generated output:
- Structure: `5599 x 57` with expected sheet/header layout.
- Skill distribution achieved:
  - Skills: `4206` (75.121%)
  - Job History: `794` (14.181%)
  - Certifications/Credentials: `297` (5.305%)
  - Language: `162` (2.893%)
  - Education: `95` (1.697%)
  - Travel: `45` (0.804%)
- Conditional leakage across controlled fields: `0` cells.
- Proficiency outside Language rows: `0`.
- Non-Yes populated Travel Interest values: `0`.
- Date-order violations (certification/job): `0`.

**Update Date: March 16, 2026 (Generator-Only Rule Alignment Refinement)**

Further updates were applied directly to `generate_synthetic_workbook.py` (no manual workbook editing):
- Smoothed per-person row allocation to prevent unrealistic concentration in a single profile.
- Added post-allocation rebalancing so one-to-many patterns remain realistic at person level.
- Tightened language-row behavior so:
  - most people have 0–2 language rows,
  - very few people have more than 2,
  - hard maximum is capped.
- Preserved strict conditional blanking so controlled columns are populated only for valid `Skill Level 1` types.
- Preserved exact global row-type distribution targets while improving profile realism.
