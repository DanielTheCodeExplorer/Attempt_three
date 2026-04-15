import ast
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from src.data_loader import parse_skills
from src.logging_utils import get_logger


LOGGER = get_logger(__name__)


def create_biography_dataset(input_csv_path: str | Path, output_csv_path: str | Path) -> pd.DataFrame:
    """Create a clean biography-based employee dataset for competency scoring."""

    input_path = Path(input_csv_path)
    output_path = Path(output_csv_path)
    LOGGER.info("create_biography_dataset_start input=%s output=%s", input_path, output_path)

    df = pd.read_csv(input_path)
    required_columns = ["TalentLink ID", "skills", "Biography"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Biography dataset source is missing required columns: {missing}")

    selected = (
        df[required_columns]
        .dropna(subset=required_columns)
        .copy()
    )
    selected["TalentLink ID"] = selected["TalentLink ID"].astype("string").str.strip()
    selected["skills"] = selected["skills"].apply(parse_skills)
    selected["biography"] = selected["Biography"].apply(parse_biography)

    selected = selected[
        selected["TalentLink ID"].notna()
        & selected["TalentLink ID"].ne("")
        & selected["skills"].map(bool)
        & selected["biography"].ne("")
    ].copy()

    grouped = (
        selected.groupby("TalentLink ID", sort=False)
        .agg(
            skills=("skills", combine_skill_lists),
            biography=("biography", combine_biographies),
        )
        .reset_index()
        .rename(columns={"TalentLink ID": "talentlinkId"})
    )

    grouped = grouped[["talentlinkId", "skills", "biography"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_path, index=False)
    LOGGER.info("create_biography_dataset_complete rows=%s output=%s", len(grouped), output_path)
    return grouped


def parse_biography(value: object) -> str:
    """Parse biography values that may be stored as a stringified list."""

    if value is None or pd.isna(value):
        return ""

    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return " ".join(parts)

    text = str(value).strip()
    if not text:
        return ""
    if text in {"<NA>", "[<NA>]", "['<NA>']", '["<NA>"]'}:
        return ""

    if text.startswith("[") and text.endswith("]"):
        try:
            literal = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return ""
        if isinstance(literal, list):
            parts = [str(item).strip() for item in literal if str(item).strip() and str(item).strip() != "<NA>"]
            return " ".join(parts)
    return text


def combine_biographies(values: pd.Series) -> str:
    """Combine biography strings into a unique ordered narrative."""

    ordered = OrderedDict()
    for value in values:
        if value:
            ordered[str(value).strip()] = None
    return " ".join(ordered.keys())


def combine_skill_lists(values: pd.Series) -> list[str]:
    """Combine repeated skill lists into a unique ordered list."""

    ordered = OrderedDict()
    for value in values:
        for skill in value:
            ordered[skill] = None
    return list(ordered.keys())
