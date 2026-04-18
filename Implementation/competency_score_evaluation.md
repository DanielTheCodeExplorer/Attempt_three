# Competency Score Evaluation

## Overview
- This note reflects the current extraction-first scoring workflow.
- The score outputs are now based on declared-skill versus extracted-skill matching, not TF-IDF cosine similarity against a fixed reference sentence.

## Current Job-History Output Summary
- Score rows evaluated: 958
- Employees evaluated: 105
- Unique skills evaluated: 10
- Average competency score: 25.95
- Average employee-level competency score: 25.95
- Matched-skill ratio: 0.2683

## Current Job-History Score Band Distribution
- low: 701 rows (73.17%)
- high: 257 rows (26.83%)

## Highest Average Skills in Job-History Output
- Stakeholder Management: avg score 97.02
- Programme Delivery: avg score 91.46
- Regulatory Reporting: avg score 65.32
- Data Analysis: avg score 4.42
- Risk Controls: avg score 4.21

## Current Biography Output Summary
- Score rows evaluated: 656
- Employees evaluated: 71
- Unique skills evaluated: 10
- Average competency score: 35.87
- Average employee-level competency score: 35.87
- Matched-skill ratio: 0.8476

## Current Biography Score Band Distribution
- medium: 513 rows (78.20%)
- low: 100 rows (15.24%)
- high: 43 rows (6.55%)

## Highest Average Skills in Biography Output
- Data Governance: avg score 44.55
- Power BI: avg score 40.00
- Regulatory Reporting: avg score 40.00
- Risk Controls: avg score 40.00
- SQL: avg score 40.00

## Interpretation
- This evaluation is descriptive rather than benchmarked against a human-labelled ground truth.
- A score now reflects whether a declared skill was extracted from the employee text and how strong that extracted evidence was.
- The score is no longer a similarity percentage. It is an evidence-support score derived from extracted skill matches.
- A `0` score means the declared skill was not supported by the extracted evidence.
- A non-zero score means the skill was supported and the score level depends on evidence strength:
  - `high -> 100`
  - `medium -> 70`
  - `low -> 40`
- The employee-level average is the mean across all declared skills for that employee.

## Methodological Note
This updated workflow is easier to explain than the earlier cosine-based benchmark comparison for the current dataset because it directly answers:

- which declared skills are supported by the text,
- and how strong that support is.

That makes the output more interpretable for audit and downstream modelling.
