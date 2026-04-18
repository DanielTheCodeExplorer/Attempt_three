from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from src.competency_scoring import CompetencyScorer, compute_tfidf_scores
from src.config import PipelineConfig, ScoreBand
from src.data_loader import (
    load_and_prepare_dataset,
    parse_skills,
    standardize_columns,
    validate_required_columns,
)
from src.text_preprocessing import preprocess_text, split_text_into_chunks


class FakeSemanticEmbedder:
    """Keyword-cluster embedder used to test the fallback path deterministically."""

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = np.array(
                [
                    self._count(lowered, {"python", "code", "coding", "script", "scripts", "automation", "automate"}),
                    self._count(lowered, {"sql", "query", "queries", "join", "joins", "database"}),
                    self._count(lowered, {"risk", "control", "controls", "assurance", "mitigation", "framework"}),
                    self._count(lowered, {"stakeholder", "stakeholders", "client", "communication", "requirements"}),
                    self._count(lowered, {"power bi", "dashboard", "dashboards", "dax", "visual"}),
                    self._count(lowered, {"programme", "program", "delivery", "milestone", "workstream"}),
                    self._count(lowered, {"regulatory", "submission", "submissions", "reporting", "compliance"}),
                    self._count(lowered, {"analysis", "analytics", "insight", "dataset", "datasets", "trend"}),
                    self._count(lowered, {"governance", "quality", "stewardship", "standards"}),
                    self._count(lowered, {"etl", "pipeline", "integration", "transform", "load", "ingestion"}),
                ],
                dtype=float,
            )
            if not vector.any():
                vector[0] = 0.1
            vectors.append(vector)
        return np.vstack(vectors)

    @staticmethod
    def _count(text: str, terms: set[str]) -> float:
        return float(sum(term in text for term in terms))


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


def test_compute_tfidf_scores_ranks_sql_above_unrelated_skills():
    profiles = {
        "SQL": "Uses SQL to query databases, join tables, and support reporting.",
        "Python": "Uses Python for scripting and automation.",
        "Stakeholder Management": "Manages stakeholders and gathers requirements.",
    }

    scores = compute_tfidf_scores(
        "Built SQL queries, created joins, and supported reporting reconciliations.",
        profiles,
        PipelineConfig(),
    )

    assert scores["SQL"] > scores["Python"]
    assert scores["SQL"] > scores["Stakeholder Management"]


def test_competency_scorer_matches_aliases_and_scores_detected_skills():
    config = PipelineConfig(
        score_bands=(
            ScoreBand(label="low", minimum_score=0.0),
            ScoreBand(label="medium", minimum_score=25.0),
            ScoreBand(label="high", minimum_score=60.0),
        )
    )
    df = pd.DataFrame(
        [
            {
                "talentlinkId": "1001",
                "description": "Built dashboard reporting in Power BI and wrote SQL queries for reconciliations.",
                "skills": ["Power BI", "SQL"],
            }
        ]
    )

    results = CompetencyScorer(config).score_dataframe(df)

    assert len(results) == 10
    required_columns = {
        "talentlinkId",
        "skill_id",
        "skill",
        "category",
        "cleaned_description",
        "source",
        "matched_aliases",
        "evidence_snippets",
        "taxonomy_score",
        "tfidf_score",
        "similarity_score",
        "graph_boost_score",
        "final_score",
        "competency_score",
        "employee_average_competency_score",
        "score_band",
    }
    assert required_columns.issubset(results.columns)

    power_bi_row = results.loc[results["skill"] == "Power BI"].iloc[0]
    sql_row = results.loc[results["skill"] == "SQL"].iloc[0]
    python_row = results.loc[results["skill"] == "Python"].iloc[0]

    assert power_bi_row["source"] == "alias"
    assert sql_row["source"] == "alias"
    assert power_bi_row["competency_score"] > 0
    assert sql_row["competency_score"] > 0
    assert json.loads(sql_row["matched_aliases"])
    assert "SQL queries" in " ".join(json.loads(sql_row["evidence_snippets"]))
    assert python_row["competency_score"] == 0.0


def test_competency_scorer_uses_embedding_fallback_when_no_alias_matches():
    config = PipelineConfig(
        minimum_output_skill_score=0.0,
        score_bands=(
            ScoreBand(label="low", minimum_score=0.0),
            ScoreBand(label="medium", minimum_score=25.0),
            ScoreBand(label="high", minimum_score=60.0),
        ),
    )
    df = pd.DataFrame(
        [
            {
                "talentlinkId": "1001",
                "description": "Developed code to automate finance checks and streamline manual reporting tasks.",
            }
        ]
    )

    results = CompetencyScorer(config, embedding_model=FakeSemanticEmbedder()).score_dataframe(df)

    python_row = results.loc[results["skill"] == "Python"].iloc[0]
    assert python_row["source"] == "embedding"
    assert python_row["taxonomy_score"] >= config.embedding_similarity_threshold
    assert python_row["competency_score"] > 0.0


def test_graph_boost_adds_small_related_scores():
    config = PipelineConfig(minimum_output_skill_score=0.0)
    df = pd.DataFrame(
        [
            {
                "talentlinkId": "1001",
                "description": "Built SQL queries, joins, and reporting extracts for month-end controls.",
            }
        ]
    )

    results = CompetencyScorer(config).score_dataframe(df)

    sql_row = results.loc[results["skill"] == "SQL"].iloc[0]
    data_analysis_row = results.loc[results["skill"] == "Data Analysis"].iloc[0]

    assert sql_row["source"] == "alias"
    assert data_analysis_row["graph_boost_score"] > 0.0
    assert data_analysis_row["competency_score"] > 0.0
    assert data_analysis_row["graph_boost_score"] < sql_row["final_score"]


def test_score_text_returns_filtered_structured_output():
    scorer = CompetencyScorer(PipelineConfig(minimum_output_skill_score=0.1))

    payload = scorer.score_text(
        "Used pandas in a Jupyter notebook to automate reporting checks and create SQL queries."
    )

    assert "skills" in payload
    assert payload["skills"]
    assert payload["skills"][0]["skill"] in {"Python", "SQL"}
    assert payload["skills"][0]["score"] >= payload["skills"][-1]["score"]
