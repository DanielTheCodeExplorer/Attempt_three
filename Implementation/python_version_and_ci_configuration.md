# Python Version and CI Configuration Update

**Update Context: Post-MVP project hardening**

This note records the configuration work completed to formalise Python version support and improve automated validation through GitHub Actions. This change was part of moving the project from a working implementation toward a more production-level structure, where environment requirements and compatibility checks are clearly defined rather than implied.

## What Was Updated

Two key project configuration files were updated:

- `pyproject.toml`
- `.github/workflows/version_requirements.yml`

These changes were made so the project declares its intended Python support explicitly and automatically checks that support through CI.

## Update to `pyproject.toml`

The `pyproject.toml` file was updated to define the project's Python compatibility range and package dependencies in a standard project configuration format.

The file now includes:

- A valid project name and version
- A short project description
- A Python version requirement:
  - `requires-python = ">=3.12,<3.15"`
- A dependency list containing:
  - `Faker`
  - `numpy`
  - `openpyxl`
  - `pandas`
  - `polars`

## Why This Change Matters

Adding `requires-python = ">=3.12,<3.15"` improves the project in several ways:

- It clearly states which Python versions the project is intended to support.
- It avoids ambiguous setup for other users or future maintainers.
- It aligns project metadata with the actual interpreter versions being tested.
- It is more production-appropriate than leaving version support undocumented.

Using a bounded range is important because it is more technically honest than claiming compatibility with all future Python releases. The configuration now states support for Python `3.12`, `3.13`, and `3.14`, without assuming that untested later versions will work automatically.

## Update to the GitHub Actions Workflow

The GitHub Actions workflow was updated so the project is no longer checked against only one Python interpreter version. Instead, the workflow now uses a version matrix in `.github/workflows/version_requirements.yml`.

The matrix currently tests:

- `3.12`
- `3.13`
- `3.14`

The workflow performs the following steps for each version:

- Checks out the repository
- Sets up the selected Python interpreter
- Installs dependencies
- Runs `flake8`
- Runs `pytest`

## Why the Workflow Change Matters

This change improves the reliability of the project configuration because the declared Python support in `pyproject.toml` is now backed by automated validation.

Previously, a single-version workflow could only confirm that the project worked in one environment. The updated matrix-based workflow is stronger because it checks the same codebase across multiple supported Python versions. This is closer to production-level practice, where compatibility claims should be supported by automated testing rather than assumption.

## Relationship Between the Two Files

The two updates are intended to work together:

- `pyproject.toml` declares the supported Python version range.
- `version_requirements.yml` tests those supported versions in CI.

This creates a clearer and more defensible project setup. The configuration no longer just says what should work; it also provides an automated mechanism to verify it.

## Summary

This update formalised Python version management in the project by defining a supported interpreter range in `pyproject.toml` and by configuring GitHub Actions to test the project across Python `3.12`, `3.13`, and `3.14`. Together, these changes make the repository more robust, more maintainable, and more aligned with production-level engineering practice.
