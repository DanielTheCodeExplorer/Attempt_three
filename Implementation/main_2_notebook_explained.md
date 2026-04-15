# `main_2.ipynb` Explanation

## Purpose
`notebooks/main_2.ipynb` is an exploratory data-preparation and feature-engineering notebook. It takes the synthetic TalentLink workbook produced by the generator, reshapes the flattened profile export into person-level analysis tables, and starts building text-based machine learning features from job-history descriptions.

In practical terms, the notebook is trying to answer:
- what skills each person has listed,
- what biography text is associated with each person,
- what job-history descriptions exist for each person,
- whether the text of those descriptions can be converted into numeric features for later modelling.

## Input
The notebook reads:

- `../data/outputs/synthetic talent link data.xlsx`

This workbook is loaded using `header=6`, which matches the report-style export layout where the real column headers begin on Excel row 7.

## What The Notebook Does

### 1. Loads and cleans the workbook
The notebook:
- reads the synthetic TalentLink Excel export into `df`,
- converts blank strings to `NaN`,
- drops fully empty rows,
- trims whitespace in string columns,
- converts `TalentLink ID` to numeric,
- parses job history date fields as datetimes.

This creates a cleaner working dataframe for grouping and text processing.

### 2. Builds a skills table per person
The notebook filters rows where `Skill Level 1 == "Skills"` and groups them by `TalentLink ID`.

Result:
- one row per person,
- a `skills` column containing a list of capability names.

This is stored in `skills_per_id_grouped`.

### 3. Builds a language table per person
The notebook separately groups rows where `Skill Level 1 == "Language"` into a list of languages per person.

Result:
- one row per person,
- a `languages` list.

This is stored in `langauage_grouped`.

At the moment, this output is exploratory only and is not joined into the final modelling table.

### 4. Builds a biography table per person
The notebook removes duplicate `TalentLink ID` and `Summary/bio` combinations, then groups by person.

Result:
- one row per person,
- a `Biography` column containing a list of summary/bio entries.

This is stored in `biography`.

### 5. Extracts and normalizes job-history rows
The notebook selects:
- `Job History Start Date`
- `Job History End Date`
- `Position`
- `Company`
- `Description`

It then:
- keeps only rows with a non-null `Description`,
- removes duplicates,
- cleans newline and repeated whitespace in text fields,
- converts dates to `YYYY-MM-DD` strings,
- sorts job entries by person and date,
- rebuilds aligned job records using `zip_longest`,
- explodes the nested job data back into a row-based dataframe.

This produces `jobs_df`, which contains one row per person-job-description combination.

### 6. Creates a combined master dataset
The notebook joins:
- grouped skills,
- grouped biographies,
- normalized job-history rows.

It then drops rows where either `skills` or `Description` is missing.

Result:
- `master_dataset`

This is the main joined analysis table used later in the notebook.

### 7. Exports the combined dataset
The notebook saves:

- `../data/outputs/master_dataset.csv`

This CSV is the cleaned joined dataset created from the notebook logic.

### 8. Creates an optional team lookup
The notebook also groups `Strategic Region` by `TalentLink ID` into a `Team` column.

Result:
- `team`

This is currently exploratory only and is not merged into the later feature-engineering pipeline.

### 9. Builds a person-level modelling dataframe
The notebook regroups `master_dataset` by `TalentLink ID` and keeps:
- first `skills`,
- first `Biography`,
- lists of job-history dates,
- lists of positions,
- lists of companies,
- lists of descriptions.

Result:
- `master_dataframe`

This is the notebook's main person-level modelling table.

### 10. Starts NLP preprocessing with spaCy
The notebook loads:

- `spacy`
- `en_core_web_sm`

It then performs:
- stop-word removal on `Description`,
- lemmatization,
- POS tagging.

Resulting columns:
- `step1_stopwords_removed`
- `step2_lemmatized`
- `pos_tags`

Important dependency note:
- this notebook requires the `en_core_web_sm` spaCy model to be installed in the active environment.

### 11. Creates TF-IDF text features
The notebook uses `TfidfVectorizer` to transform lemmatized description text into numeric vectors.

It creates:
- a per-row TF-IDF matrix,
- a grouped per-person TF-IDF matrix.

This is the first step toward converting profile text into machine-learning-ready features.

### 12. Scores listed skills against description text
The notebook then:
- groups employee text by `TalentLink ID`,
- vectorizes it again with TF-IDF,
- checks each employee's listed skills against TF-IDF feature names,
- assigns a score when a listed skill appears in the employee text.

Result:
- `skill_scores_df`

After that, it filters to positive matches only and builds:
- `skill_analysis_table`

This final table summarizes:
- how many employees demonstrate each skill in text,
- average TF-IDF score,
- maximum TF-IDF score.

## Current Output Files
The notebook currently writes:

- `../data/outputs/master_dataset.csv`

Most other outputs remain in memory as notebook dataframes rather than being saved as separate files.

## Current Limitations
The notebook is exploratory and not yet fully productionized. Current limitations include:

- `languages` and `team` are created but not integrated into the final modelling dataframe.
- `Biography` is collected but not yet included in the TF-IDF and skill-scoring logic.
- The notebook depends on the external spaCy model `en_core_web_sm`.
- There is repeated NLP setup code that could be consolidated.
- The workflow is notebook-driven rather than packaged as reusable functions or scripts.

## Short Summary
`main_2.ipynb` takes the flattened synthetic TalentLink export, reshapes it into person-level analysis tables, exports a cleaned master dataset, and begins transforming job-description text into machine learning features using spaCy and TF-IDF.
