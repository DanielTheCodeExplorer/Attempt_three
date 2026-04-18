import json
from pathlib import Path

import pandas as pd

from src.config import PipelineConfig
from src.final_competency_dataset import build_final_competency_dataset


def test_build_final_competency_dataset_creates_consolidated_file(tmp_path: Path):
    input_df = pd.DataFrame(
        [
            {
                "TalentLink ID": "1001",
                "skills": "['Python', 'SQL']",
                "Biography": "Built Python automation for reporting.",
                "Job History Start Date": "2020-01-01",
                "Job History End Date": "2021-01-01",
                "Position": "Consultant",
                "Company": "ACME",
                "Description": "Created SQL queries for reconciliations.",
            }
        ]
    )
    input_path = tmp_path / "master.csv"
    output_path = tmp_path / "final.csv"
    input_df.to_csv(input_path, index=False)

    config = PipelineConfig(
        input_csv_path=input_path,
        biography_input_csv_path=input_path,
        final_competency_dataset_csv_path=output_path,
    )

    final_df = build_final_competency_dataset(config)

    assert output_path.exists()
    assert len(final_df) == 10
    assert set(
        [
            "talentlinkId",
            "skill",
            "job_description_competency_score",
            "biography_competency_score",
            "job_description_tfidf_score",
            "biography_tfidf_score",
            "job_description_taxonomy_score",
            "biography_taxonomy_score",
            "job_description_graph_boost_score",
            "biography_graph_boost_score",
            "job_description_final_score",
            "biography_final_score",
            "job_description_confidence_score",
            "biography_confidence_score",
            "confidence_score",
            "sum_tfidf_score",
            "sum_competency_score",
            "competency_score_rank",
            "declared_skills_json",
            "is_declared_skill",
        ]
    ).issubset(final_df.columns)
    python_row = final_df.loc[final_df["skill"] == "Python"].iloc[0]
    sql_row = final_df.loc[final_df["skill"] == "SQL"].iloc[0]

    assert bool(python_row["is_declared_skill"]) is True
    assert bool(sql_row["is_declared_skill"]) is True
    assert "Python" in json.loads(python_row["declared_skills_json"])
    assert python_row["biography_competency_score"] >= 0
    assert sql_row["job_description_competency_score"] >= 0
    assert python_row["biography_taxonomy_score"] > 0
    assert sql_row["job_description_taxonomy_score"] > 0
    assert python_row["biography_final_score"] >= python_row["biography_tfidf_score"]
    assert sql_row["job_description_final_score"] >= sql_row["job_description_tfidf_score"]
    assert python_row["confidence_score"] >= 0
    assert sql_row["sum_tfidf_score"] >= 0
    assert sql_row["sum_competency_score"] >= 0
    zero_rows = final_df.loc[final_df["sum_competency_score"] == 0]
    if not zero_rows.empty:
        assert set(zero_rows["competency_score_rank"]) == {-1}
