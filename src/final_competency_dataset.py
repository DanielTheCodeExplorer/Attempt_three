import json

import pandas as pd

from src.biography_dataset import create_biography_dataset
from src.competency_scoring import CompetencyScorer, compute_tfidf_scores
from src.config import PipelineConfig
from src.data_loader import load_and_prepare_dataset
from src.llm_normalizer import extract_supported_skills_from_text
from src.logging_utils import get_logger
from src.skill_taxonomy import build_default_skill_taxonomy


LOGGER = get_logger(__name__)

CONFIDENCE_SCORE_MAP = {
    "high": 1.0,
    "medium": 0.7,
    "low": 0.4,
}


def build_final_competency_dataset(config: PipelineConfig) -> pd.DataFrame:
    """Create a final analysis dataset with side-by-side biography and job-history scores."""

    taxonomy = build_default_skill_taxonomy()
    scorer = CompetencyScorer(config, taxonomy=taxonomy)
    canonical_skills = [skill.canonical_name for skill in taxonomy.skills]
    skill_profiles = taxonomy.skill_profiles()

    job_history_df = load_and_prepare_dataset(config.input_csv_path, config)
    biography_df = create_biography_dataset(config.biography_input_csv_path, output_csv_path=None)
    base_df = _build_employee_base_frame(job_history_df, biography_df)

    records: list[dict[str, object]] = []
    for _, row in base_df.iterrows():
        biography_tfidf = compute_tfidf_scores(row["biography_text"], skill_profiles, config)
        job_description_tfidf = compute_tfidf_scores(row["job_history_text"], skill_profiles, config)
        biography_taxonomy = _build_taxonomy_lookup(row["biography_text"], scorer)
        job_description_taxonomy = _build_taxonomy_lookup(row["job_history_text"], scorer)
        biography_confidence = _build_confidence_lookup(row["biography_text"], canonical_skills, config)
        job_description_confidence = _build_confidence_lookup(row["job_history_text"], canonical_skills, config)

        for skill in taxonomy.skills:
            biography_taxonomy_payload = biography_taxonomy.get(skill.skill_id, _empty_taxonomy_payload())
            job_description_taxonomy_payload = job_description_taxonomy.get(skill.skill_id, _empty_taxonomy_payload())
            biography_confidence_score = float(biography_confidence.get(skill.canonical_name, 0.0))
            job_description_confidence_score = float(job_description_confidence.get(skill.canonical_name, 0.0))
            biography_tfidf_score = float(biography_tfidf.get(skill.canonical_name, 0.0))
            job_description_tfidf_score = float(job_description_tfidf.get(skill.canonical_name, 0.0))
            biography_taxonomy_score = float(biography_taxonomy_payload["taxonomy_score"])
            job_description_taxonomy_score = float(job_description_taxonomy_payload["taxonomy_score"])
            biography_graph_boost_score = float(biography_taxonomy_payload["graph_boost_score"])
            job_description_graph_boost_score = float(job_description_taxonomy_payload["graph_boost_score"])
            biography_final_score = float(biography_taxonomy_payload["final_score"])
            job_description_final_score = float(job_description_taxonomy_payload["final_score"])
            confidence_score = ((biography_confidence_score + job_description_confidence_score) / 2) * 100
            sum_tfidf_score = biography_tfidf_score + job_description_tfidf_score
            biography_competency_score = biography_final_score * biography_confidence_score * 100
            job_description_competency_score = job_description_final_score * job_description_confidence_score * 100
            sum_competency_score = biography_competency_score + job_description_competency_score

            records.append(
                {
                    "talentlinkId": row["talentlinkId"],
                    "skill_id": skill.skill_id,
                    "skill": skill.canonical_name,
                    "category": skill.category,
                    "declared_skills_json": json.dumps(row["declared_skills"], ensure_ascii=True),
                    "is_declared_skill": skill.canonical_name in row["declared_skills"],
                    "position_text": row["position_text"],
                    "job_history_text": row["job_history_text"],
                    "biography_text": row["biography_text"],
                    "combined_text": row["combined_text"],
                    "biography_tfidf_score": round(biography_tfidf_score, 6),
                    "job_description_tfidf_score": round(job_description_tfidf_score, 6),
                    "biography_taxonomy_score": round(biography_taxonomy_score, 6),
                    "job_description_taxonomy_score": round(job_description_taxonomy_score, 6),
                    "biography_graph_boost_score": round(biography_graph_boost_score, 6),
                    "job_description_graph_boost_score": round(job_description_graph_boost_score, 6),
                    "biography_final_score": round(biography_final_score, 6),
                    "job_description_final_score": round(job_description_final_score, 6),
                    "biography_confidence_score": round(biography_confidence_score, 6),
                    "job_description_confidence_score": round(job_description_confidence_score, 6),
                    "confidence_score": round(confidence_score, 2),
                    "sum_tfidf_score": round(sum_tfidf_score, 6),
                    "biography_competency_score": round(biography_competency_score, 2),
                    "job_description_competency_score": round(job_description_competency_score, 2),
                    "sum_competency_score": round(sum_competency_score, 2),
                }
            )

    final_df = pd.DataFrame(records).sort_values(["talentlinkId", "skill"]).reset_index(drop=True)
    final_df["competency_score_rank"] = (
        final_df.groupby("talentlinkId")["sum_competency_score"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    final_df.loc[final_df["sum_competency_score"] == 0, "competency_score_rank"] = -1
    config.final_competency_dataset_csv_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(config.final_competency_dataset_csv_path, index=False)
    LOGGER.info(
        "final_competency_dataset_complete output=%s rows=%s employees=%s",
        config.final_competency_dataset_csv_path,
        len(final_df),
        final_df["talentlinkId"].nunique(),
    )
    return final_df


def _build_employee_base_frame(job_history_df: pd.DataFrame, biography_df: pd.DataFrame) -> pd.DataFrame:
    biography_view = biography_df.rename(
        columns={
            "skills": "biography_skills",
            "biography": "biography_text",
            "description": "biography_source_description",
            "position": "position_text",
        }
    )
    merged = job_history_df.rename(columns={"description": "job_history_text", "skills": "job_history_skills"}).merge(
        biography_view,
        on="talentlinkId",
        how="outer",
    )

    merged["job_history_text"] = merged["job_history_text"].fillna("").astype(str)
    merged["biography_text"] = merged["biography_text"].fillna("").astype(str)
    merged["position_text"] = merged["position_text"].fillna("").astype(str)
    merged["job_history_skills"] = merged["job_history_skills"].apply(_ensure_skill_list)
    merged["biography_skills"] = merged["biography_skills"].apply(_ensure_skill_list)
    merged["declared_skills"] = merged.apply(
        lambda row: _combine_skill_lists(row["job_history_skills"], row["biography_skills"]),
        axis=1,
    )
    merged["combined_text"] = merged.apply(
        lambda row: " ".join(
            part.strip()
            for part in [row["biography_text"], row["job_history_text"]]
            if str(part).strip()
        ),
        axis=1,
    )
    return merged[
        [
            "talentlinkId",
            "declared_skills",
            "job_history_text",
            "biography_text",
            "combined_text",
            "position_text",
        ]
    ].copy()


def _build_confidence_lookup(text: str, canonical_skills: list[str], config: PipelineConfig) -> dict[str, float]:
    payload = extract_supported_skills_from_text(
        description_text=text,
        allowed_skills=canonical_skills,
        config=config,
    )
    lookup: dict[str, float] = {}
    for item in payload.get("matched_skills", []):
        skill = str(item.get("skill", "")).strip()
        evidence_strength = str(item.get("evidence_strength", "low")).strip().lower()
        if skill and evidence_strength in CONFIDENCE_SCORE_MAP:
            lookup[skill] = CONFIDENCE_SCORE_MAP[evidence_strength]
    return lookup


def _build_taxonomy_lookup(text: str, scorer: CompetencyScorer) -> dict[str, dict[str, object]]:
    return scorer._score_all_skills(text)


def _empty_taxonomy_payload() -> dict[str, object]:
    return {
        "taxonomy_score": 0.0,
        "graph_boost_score": 0.0,
        "final_score": 0.0,
    }


def _ensure_skill_list(value: object) -> list[str]:
    if isinstance(value, list):
        return value
    return []


def _combine_skill_lists(*skill_lists: list[str]) -> list[str]:
    ordered: dict[str, None] = {}
    for skill_list in skill_lists:
        for skill in skill_list:
            ordered[str(skill)] = None
    return list(ordered.keys())
