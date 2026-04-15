from pathlib import Path

import pandas as pd

from src.biography_dataset import create_biography_dataset, parse_biography
from src.config import PipelineConfig
from src.data_loader import load_and_prepare_dataset


def test_parse_biography_handles_stringified_list():
    value = "['Experienced in risk and data delivery.', 'Supports reporting improvements.']"

    assert parse_biography(value) == "Experienced in risk and data delivery. Supports reporting improvements."


def test_create_biography_dataset_filters_na_and_groups_rows(tmp_path: Path):
    source_df = pd.DataFrame(
        [
            {
                "TalentLink ID": "1001",
                "skills": "['Python', 'SQL']",
                "Biography": "['Experienced in Python automation.']",
            },
            {
                "TalentLink ID": "1001",
                "skills": "['SQL', 'Python']",
                "Biography": "['Experienced in Python automation.']",
            },
            {
                "TalentLink ID": "1002",
                "skills": None,
                "Biography": "['Missing skill row.']",
            },
            {
                "TalentLink ID": "1003",
                "skills": "['Power BI']",
                "Biography": None,
            },
        ]
    )
    input_path = tmp_path / "master_dataset.csv"
    output_path = tmp_path / "biography_dataset.csv"
    source_df.to_csv(input_path, index=False)

    result = create_biography_dataset(input_path, output_path)

    assert output_path.exists()
    assert list(result.columns) == ["talentlinkId", "skills", "biography"]
    assert result.shape[0] == 1
    assert result.loc[0, "talentlinkId"] == "1001"
    assert result.loc[0, "skills"] == ["Python", "SQL"]
    assert result.loc[0, "biography"] == "Experienced in Python automation."


def test_biography_dataset_loads_into_scoring_schema(tmp_path: Path):
    biography_df = pd.DataFrame(
        [
            {
                "talentlinkId": "1001",
                "skills": "['Python', 'SQL']",
                "biography": "Experienced in Python automation and SQL reporting.",
            }
        ]
    )
    csv_path = tmp_path / "biography_dataset.csv"
    biography_df.to_csv(csv_path, index=False)

    prepared = load_and_prepare_dataset(csv_path, PipelineConfig())

    assert list(prepared.columns) == ["talentlinkId", "description", "skills"]
    assert prepared.loc[0, "description"] == "Experienced in Python automation and SQL reporting."
    assert prepared.loc[0, "skills"] == ["Python", "SQL"]
