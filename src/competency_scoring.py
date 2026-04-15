from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import PipelineConfig, ScoreBand
from src.logging_utils import get_logger
from src.skill_references import get_skill_reference
from src.text_preprocessing import extract_evidence_excerpt, preprocess_text


LOGGER = get_logger(__name__)


@dataclass
class ScoreRecord:
    """Represents one employee-skill audit row."""

    talentlinkId: str
    skill: str
    cleaned_description: str
    reference_text: str
    similarity_score: float
    competency_score: float
    score_band: str
    evidence_excerpt: str


class CompetencyScorer:
    """Scores listed skills against employee description evidence."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=config.tfidf_max_features,
            ngram_range=config.tfidf_ngram_range,
        )

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate an explainable skill score for each employee-skill pair."""

        self._validate_input_frame(df)
        working = df.copy()
        working["cleaned_description"] = working["description"].apply(preprocess_text)

        reference_lookup = self._build_reference_lookup(working["skills"])
        self._fit_vectorizer(working["cleaned_description"], reference_lookup)

        description_vectors = self.vectorizer.transform(working["cleaned_description"])
        reference_vectors = {
            skill: self.vectorizer.transform([reference_text])
            for skill, reference_text in reference_lookup.items()
        }

        records: list[ScoreRecord] = []
        for row_index, row in working.reset_index(drop=True).iterrows():
            description_vector = description_vectors[row_index]
            for skill in row["skills"]:
                reference_text = reference_lookup[skill]
                similarity = float(cosine_similarity(description_vector, reference_vectors[skill])[0][0])
                records.append(
                    ScoreRecord(
                        talentlinkId=row["talentlinkId"],
                        skill=skill,
                        cleaned_description=row["cleaned_description"],
                        reference_text=reference_text,
                        similarity_score=round(similarity, 6),
                        competency_score=self._to_competency_score(similarity),
                        score_band=self._assign_score_band(similarity),
                        evidence_excerpt=extract_evidence_excerpt(
                            description=row["description"],
                            reference_text=reference_text,
                            skill=skill,
                        ),
                    )
                )

        LOGGER.info("score_dataframe_complete employees=%s scores=%s", len(working), len(records))
        return pd.DataFrame([record.__dict__ for record in records])

    def _validate_input_frame(self, df: pd.DataFrame) -> None:
        required = {"talentlinkId", "description", "skills"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Scoring dataframe is missing required columns: {sorted(missing)}")

    def _build_reference_lookup(self, skill_series: pd.Series) -> dict[str, str]:
        unique_skills: dict[str, str] = {}
        for skills in skill_series:
            for skill in skills:
                if skill not in unique_skills:
                    unique_skills[skill] = get_skill_reference(skill, self.config)
        return unique_skills

    def _fit_vectorizer(self, cleaned_descriptions: pd.Series, reference_lookup: dict[str, str]) -> None:
        corpus = list(cleaned_descriptions) + list(reference_lookup.values())
        self.vectorizer.fit(corpus)

    def _to_competency_score(self, similarity: float) -> float:
        return round(similarity * 100, 2)

    def _assign_score_band(self, similarity: float) -> str:
        selected_band: ScoreBand = self.config.score_bands[0]
        for band in self.config.score_bands:
            if similarity >= band.minimum_similarity:
                selected_band = band
        return selected_band.label
