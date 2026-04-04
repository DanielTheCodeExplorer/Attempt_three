# Folder Restructure After MVP Acceptance

**Restructure Date Context: After MVP acceptance**

This note explains how the project was restructured from the earlier MVP-stage layout in the git history to the current layout now used in the working project. The restructure was completed after the MVP had been accepted, at the point where the focus moved from proving that the generator worked to organising it in a way that is more suitable for ongoing development, maintenance, and production-style delivery.

## Why the Restructure Was Done

The earlier version of the project was appropriate for MVP delivery because it demonstrated the core technical outcome quickly: generating synthetic workbook data from the profiled schema. However, once the MVP was accepted, the project needed to move beyond a prototype layout and into a structure that separates code, inputs, outputs, tests, and implementation evidence more clearly.

This change reflects a shift from an experimentation-first structure to a more production-level codebase layout. In other words, the generator was no longer being treated as a single script surrounded by ad hoc files, but as a project with clearer boundaries between source logic, data assets, generated artefacts, and supporting documentation.

## Previous Folder Structure in the Older Git History

In the earlier git-tracked version, the project was largely flat and prototype-oriented. Key characteristics included:

- The main generator script was stored inside `generating synthetic results/`.
- Input files such as the schema and example workbook were kept alongside generated outputs.
- Multiple generated Excel outputs were stored in the same area as the implementation script.
- The notebook sat at the project root.
- There was no clear separation between source code, raw inputs, generated deliverables, and test material.

Examples from the older history include:

- `generating synthetic results/generate_synthetic_workbook.py`
- `generating synthetic results/schema_profile.json`
- `generating synthetic results/synthetic_staff_talentlink_v*.xlsx`
- `main_2.ipynb`

That structure worked for MVP purposes because it kept everything close together during rapid iteration, but it was less suitable for long-term maintainability. It mixed operational artefacts with implementation logic and made the project harder to navigate as the number of files increased.

## Current Folder Structure

The current version has been reorganised into clearer project areas:

- `src/` for implementation code
- `data/inputs/` for source inputs and reference files
- `data/outputs/` for generated outputs
- `tests/` for validation work
- `Implementation/` for written evidence and technical review

Examples from the current structure include:

- `src/generate_synthetic_workbook.py`
- `data/inputs/schema_profile.json`
- `data/inputs/Dummy_Profile_Examples.xlsx`
- `data/outputs/master_dataset.csv`
- `data/outputs/talentlink_versions/`
- `tests/test_dummy_data.py`
- `Implementation/generating_dummy_data_implementation_review.md`

## What Changed Between the Old Version and the New Version

The main restructuring changes were:

1. The generator script was moved out of a results folder and placed into `src/`.
This makes the codebase easier to understand because the implementation now lives in a dedicated source directory instead of being mixed with output files.

2. Input data and generated output were separated.
Schema files, example workbooks, and other reference inputs are now stored under `data/inputs/`, while generated artefacts are stored under `data/outputs/`. This is a more controlled and maintainable way to manage data flow.

3. Generated workbook versions were grouped more cleanly.
Instead of keeping workbook versions next to the script, output versions are now collected under `data/outputs/talentlink_versions/`, which makes versioned deliverables easier to track.

4. Testing and project support files were introduced into the structure.
The addition of `tests/`, `README.md`, `pyproject.toml`, and licensing/supporting project files reflects a move toward a more formal software project layout rather than a single-script prototype.

5. Documentation and implementation evidence were kept separate from code.
The `Implementation/` folder now acts as the location for written project evidence, which supports traceability and professional presentation of the work completed.

## Why the New Structure Is More Production-Level

The new structure is more production-level because it improves:

- Maintainability: code, data, and outputs are easier to locate and update.
- Readability: the purpose of each directory is clearer to another developer or reviewer.
- Scalability: new scripts, tests, inputs, and output versions can be added without cluttering the root of the repository.
- Traceability: implementation evidence is separated from executable code and generated files.
- Workflow discipline: the project now follows a structure closer to standard software engineering practice, where source code is not stored inside the same folder as generated artefacts.

This does not just make the folder tree look cleaner. It changes the project from an MVP/prototype arrangement into one that is better suited to continued development, review, testing, and handover.

## Summary

After the MVP was accepted, the project was restructured to support a more production-style workflow. The older version in the git history was effective for rapid development and demonstrating the core generator, but it combined scripts, inputs, and outputs in a single results-focused layout. The new version separates implementation code, inputs, outputs, tests, and documentation into dedicated areas, making the project more maintainable, more professional, and more aligned with production-level engineering practice.
