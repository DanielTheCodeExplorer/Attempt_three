from pathlib import Path

import pandas as pd
import pytest

from src.competency_scoring import CompetencyScorer
from src.config import PipelineConfig, ScoreBand
from src.data_loader import (
    load_and_prepare_dataset,
    parse_skills,
    standardize_columns,
    validate_required_columns,
)
from src.text_preprocessing import preprocess_text


def test_validate_required_columns_rejects_missing_fields():
    df = pd.DataFrame({"talentlinkId": ["1001"], "description": ["Python developer"]})

    with pytest.raises(ValueError, match="skills"):
        validate_required_columns(df, PipelineConfig())


def test_parse_skills_handles_stringified_lists_and_deduplicates():
    assert parse_skills("['Python', 'SQL', 'Python']") == ["Python", "SQL"]
    assert parse_skills("Python, SQL, Python") == ["Python", "SQL"]


def test_preprocess_text_normalises_noise_but_keeps_terms():
    text = " Python, automation & DATA-analysis!! "

    assert preprocess_text(text) == "python automation data analysis"


def test_load_and_prepare_dataset_standardises_and_aggregates_rows(tmp_path: Path):
    input_df = pd.DataFrame(
        [
            {
                "TalentLink ID": "1001",
                "Description": "Built Python automations.",
                "skills": "['Python', 'SQL']",
            },
            {
                "TalentLink ID": "1001",
                "Description": "Created SQL reports.",
                "skills": "['SQL', 'Python']",
            },
            {
                "TalentLink ID": "1002",
                "Description": None,
                "skills": "['Power BI']",
            },
        ]
    )
    csv_path = tmp_path / "input.csv"
    input_df.to_csv(csv_path, index=False)

    prepared = load_and_prepare_dataset(csv_path, PipelineConfig())

    assert list(prepared.columns) == ["talentlinkId", "description", "skills"]
    assert prepared.shape[0] == 1
    assert prepared.loc[0, "talentlinkId"] == "1001"
    assert prepared.loc[0, "skills"] == ["Python", "SQL"]
    assert "Built Python automations." in prepared.loc[0, "description"]
    assert "Created SQL reports." in prepared.loc[0, "description"]


def test_standardize_columns_does_not_create_duplicate_description_columns():
    df = pd.DataFrame(
        [
            {
                "TalentLink ID": "1001",
                "Description": "Job history description",
                "Biography": "Biography text",
                "skills": "['Python']",
            }
        ]
    )

    standardized = standardize_columns(df, PipelineConfig())

    assert standardized.columns.tolist().count("description") == 1
    assert standardized.loc[0, "description"] == "Job history description"


def test_competency_scorer_generates_audit_trail_and_relative_scores():
    config = PipelineConfig(
        tfidf_max_features=None,
        score_bands=(
            ScoreBand(label="low", minimum_similarity=0.0),
            ScoreBand(label="medium", minimum_similarity=0.1),
            ScoreBand(label="high", minimum_similarity=0.3),
        ),
        skill_references={
            "python": "Uses Python for scripting automation and data analysis.",
            "sql": "Uses SQL to query tables and write reporting joins.",
        },
    )
    df = pd.DataFrame(
        [
            {
                "talentlinkId": "1001",
                "description": "Python scripting automation and data analysis for reporting.",
                "skills": ["Python", "SQL"],
            }
        ]
    )

    results = CompetencyScorer(config).score_dataframe(df)

    assert set(
        [
            "talentlinkId",
            "skill",
            "cleaned_description",
            "reference_text",
            "similarity_score",
            "competency_score",
            "score_band",
            "evidence_excerpt",
        ]
    ).issubset(results.columns)
    assert len(results) == 2

    python_score = results.loc[results["skill"] == "Python", "similarity_score"].iloc[0]
    sql_score = results.loc[results["skill"] == "SQL", "similarity_score"].iloc[0]

    assert python_score > sql_score
    assert results.loc[results["skill"] == "Python", "score_band"].iloc[0] in {"medium", "high"}
