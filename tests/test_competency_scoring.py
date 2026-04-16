from pathlib import Path
import json

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
from src.text_preprocessing import preprocess_text, split_text_into_chunks


class FakeSkillRewriteClient:
    class chat:
        class completions:
            @staticmethod
            def create(*args, **kwargs):
                user_message = kwargs["messages"][1]["content"].lower()
                if "target skill:\npython" in user_message:
                    payload = {
                        "skill_focused_description": "python scripting automation and data analysis for reporting",
                        "evidence_strength": "high",
                    }
                else:
                    payload = {
                        "skill_focused_description": "planned catering budgets for office events",
                        "evidence_strength": "low",
                    }
                response = type(
                    "Response",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {"message": type("Message", (), {"content": json.dumps(payload)})()},
                            )()
                        ]
                    },
                )()
                return response


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


def test_split_text_into_chunks_breaks_text_into_sentences():
    text = "Built dashboards for reporting. Queried customer data using SQL."

    assert split_text_into_chunks(text) == [
        "Built dashboards for reporting.",
        "Queried customer data using SQL.",
    ]


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
            ScoreBand(label="low", minimum_score=0.0),
            ScoreBand(label="medium", minimum_score=35.0),
            ScoreBand(label="high", minimum_score=65.0),
        ),
        exact_match_weight=0.6,
        semantic_similarity_weight=0.4,
        skill_references={
            "python": "Uses Python for scripting automation and data analysis.",
            "sql": "Uses SQL to query tables and write reporting joins.",
        },
        skill_aliases={
            "python": ("pandas",),
            "sql": ("relational databases", "joins"),
        },
    )
    df = pd.DataFrame(
        [
            {
                "talentlinkId": "1001",
                "description": "Planned catering budgets for office events. Python scripting automation and data analysis for reporting.",
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
            "match_signal_score",
            "similarity_score",
            "semantic_evidence_score",
            "hybrid_score",
            "competency_score",
            "score_band",
            "evidence_excerpt",
        ]
    ).issubset(results.columns)
    assert len(results) == 2

    python_score = results.loc[results["skill"] == "Python", "similarity_score"].iloc[0]
    sql_score = results.loc[results["skill"] == "SQL", "similarity_score"].iloc[0]
    python_competency = results.loc[results["skill"] == "Python", "competency_score"].iloc[0]

    assert python_score > sql_score
    assert python_competency >= 60
    assert results.loc[results["skill"] == "Python", "score_band"].iloc[0] == "high"


def test_alias_matches_produce_moderate_or_better_scores():
    config = PipelineConfig(
        tfidf_max_features=None,
        score_bands=(
            ScoreBand(label="low", minimum_score=0.0),
            ScoreBand(label="medium", minimum_score=25.0),
            ScoreBand(label="high", minimum_score=60.0),
        ),
        skill_references={
            "sql": "Uses SQL to query relational databases and write joins for reporting.",
        },
        skill_aliases={
            "sql": ("relational databases", "joins"),
        },
    )
    df = pd.DataFrame(
        [
            {
                "talentlinkId": "1001",
                "description": "Analysed customer records and extracted data from relational databases with complex joins.",
                "skills": ["SQL"],
            }
        ]
    )

    results = CompetencyScorer(config).score_dataframe(df)

    assert results.loc[0, "match_signal_score"] == 0.8
    assert results.loc[0, "competency_score"] >= 35.0
    assert results.loc[0, "score_band"] in {"medium", "high"}


def test_semantic_similarity_is_calibrated_before_hybrid_scoring():
    config = PipelineConfig(
        tfidf_max_features=None,
        semantic_similarity_strong_threshold=0.10,
        exact_match_weight=0.6,
        semantic_similarity_weight=0.4,
    )
    scorer = CompetencyScorer(config)

    assert scorer._calibrate_similarity(0.05) == 0.5
    assert scorer._calibrate_similarity(0.20) == 1.0


def test_competency_scorer_can_use_skill_focused_rewrite_per_skill():
    config = PipelineConfig(
        tfidf_max_features=None,
        skill_references={
            "python": "Uses Python for scripting automation and data analysis.",
            "sql": "Uses SQL to query tables and write reporting joins.",
        },
    )
    df = pd.DataFrame(
        [
            {
                "talentlinkId": "1001",
                "description": "Planned catering budgets for office events. Python scripting automation and data analysis for reporting.",
                "skills": ["Python", "SQL"],
            }
        ]
    )

    results = CompetencyScorer(config, llm_client=FakeSkillRewriteClient()).score_dataframe(df)

    python_row = results.loc[results["skill"] == "Python"].iloc[0]
    sql_row = results.loc[results["skill"] == "SQL"].iloc[0]

    assert python_row["cleaned_description"] == "python scripting automation and data analysis for reporting"
    assert sql_row["cleaned_description"] == "planned catering budgets for office events"
    assert python_row["similarity_score"] > sql_row["similarity_score"]
