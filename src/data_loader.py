import ast
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from src.config import PipelineConfig
from src.logging_utils import get_logger


LOGGER = get_logger(__name__)


def load_and_prepare_dataset(path: str | Path, config: PipelineConfig) -> pd.DataFrame:
    """Load a CSV, validate required fields, and collapse it to one row per employee."""

    csv_path = Path(path)
    LOGGER.info("load_dataset_start path=%s", csv_path)
    df = pd.read_csv(csv_path)
    standardized = standardize_columns(df, config)
    validate_required_columns(standardized, config)

    prepared = standardized.copy()
    prepared["talentlinkId"] = prepared["talentlinkId"].astype("string").str.strip()
    prepared["description"] = prepared["description"].apply(normalise_description_value)
    prepared["skills"] = prepared["skills"].apply(parse_skills)

    prepared = prepared[
        prepared["talentlinkId"].notna()
        & prepared["description"].ne("")
        & prepared["skills"].map(bool)
    ].copy()

    grouped = (
        prepared.groupby("talentlinkId", sort=False)
        .agg(
            description=("description", combine_descriptions),
            skills=("skills", combine_skill_lists),
        )
        .reset_index()
    )

    LOGGER.info(
        "load_dataset_complete employees=%s rows=%s",
        grouped["talentlinkId"].nunique(),
        len(grouped),
    )
    return grouped


def standardize_columns(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Rename known aliases to the canonical field names used by the pipeline."""

    rename_map: dict[str, str] = {}
    claimed_targets = set(df.columns)
    for source, target in config.input_column_aliases.items():
        if source not in df.columns:
            continue
        if target in claimed_targets:
            continue
        rename_map[source] = target
        claimed_targets.add(target)
    return df.rename(columns=rename_map)


def validate_required_columns(df: pd.DataFrame, config: PipelineConfig) -> None:
    """Ensure the dataframe contains the required canonical fields."""

    missing = [column for column in config.required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def parse_skills(value: object) -> list[str]:
    """Parse skill data from a Python-list string, list object, or comma-separated text."""

    if value is None or pd.isna(value):
        return []

    parsed: list[object]
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            literal = ast.literal_eval(text)
            if not isinstance(literal, list):
                raise ValueError(f"Skills value is not a list: {value}")
            parsed = literal
        else:
            parsed = text.split(",")
    else:
        raise ValueError(f"Unsupported skills value: {value!r}")

    ordered = OrderedDict()
    for skill in parsed:
        normalized = str(skill).strip().strip("'\"")
        if normalized:
            ordered[normalized] = None
    return list(ordered.keys())


def normalise_description_value(value: object) -> str:
    """Convert raw description values into a safe string."""

    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def combine_descriptions(values: pd.Series) -> str:
    """Combine repeated employee descriptions into one narrative string."""

    ordered = OrderedDict()
    for value in values:
        normalized = normalise_description_value(value)
        if normalized:
            ordered[normalized] = None
    return " ".join(ordered.keys())


def combine_skill_lists(values: pd.Series) -> list[str]:
    """Combine repeated employee skill lists into a unique ordered list."""

    ordered = OrderedDict()
    for value in values:
        for skill in value:
            ordered[skill] = None
    return list(ordered.keys())
