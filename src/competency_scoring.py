from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import PipelineConfig, ScoreBand
from src.logging_utils import get_logger
from src.skill_references import get_skill_aliases, get_skill_reference
from src.text_preprocessing import preprocess_text, split_text_into_chunks


LOGGER = get_logger(__name__)


@dataclass
class ScoreRecord:
    """Represents one employee-skill audit row."""

    talentlinkId: str
    skill: str
    cleaned_description: str
    reference_text: str
    match_signal_score: float
    similarity_score: float
    semantic_evidence_score: float
    hybrid_score: float
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
            stop_words=config.tfidf_stop_words,
            sublinear_tf=config.tfidf_sublinear_tf,
        )

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate an explainable skill score for each employee-skill pair."""

        self._validate_input_frame(df)
        working = df.copy()
        working["cleaned_description"] = working["description"].apply(preprocess_text)
        working["description_chunks"] = working["description"].apply(split_text_into_chunks)

        reference_lookup = self._build_reference_lookup(working["skills"])
        self._fit_vectorizer(working["description_chunks"], reference_lookup)
        reference_vectors = self._build_reference_vectors(reference_lookup)

        records: list[ScoreRecord] = []
        for row_index, row in working.reset_index(drop=True).iterrows():
            for skill in row["skills"]:
                reference_text = reference_lookup[skill]
                chunk_similarity, evidence_excerpt = self._best_chunk_similarity(
                    row["description_chunks"], reference_vectors[skill]
                )
                match_signal = self._compute_match_signal(row["cleaned_description"], skill)
                semantic_evidence_score = self._calibrate_similarity(chunk_similarity)
                hybrid_score = self._combine_signals(match_signal, semantic_evidence_score)
                competency_score = self._to_competency_score(hybrid_score)
                records.append(
                    ScoreRecord(
                        talentlinkId=row["talentlinkId"],
                        skill=skill,
                        cleaned_description=row["cleaned_description"],
                        reference_text=reference_text,
                        match_signal_score=round(match_signal, 6),
                        similarity_score=round(chunk_similarity, 6),
                        semantic_evidence_score=round(semantic_evidence_score, 6),
                        hybrid_score=round(hybrid_score, 6),
                        competency_score=competency_score,
                        score_band=self._assign_score_band(competency_score),
                        evidence_excerpt=evidence_excerpt,
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

    def _fit_vectorizer(self, description_chunks: pd.Series, reference_lookup: dict[str, str]) -> None:
        corpus = []
        for chunks in description_chunks:
            corpus.extend(preprocess_text(chunk) for chunk in chunks if preprocess_text(chunk))
        corpus.extend(reference_lookup.values())
        self.vectorizer.fit(corpus)

    def _build_reference_vectors(self, reference_lookup: dict[str, str]) -> dict[str, object]:
        return {
            skill: self.vectorizer.transform([preprocess_text(reference_text)])
            for skill, reference_text in reference_lookup.items()
        }

    def _best_chunk_similarity(self, chunks: list[str], reference_vector: object) -> tuple[float, str]:
        best_similarity = 0.0
        best_chunk = ""
        for chunk in chunks:
            cleaned_chunk = preprocess_text(chunk)
            if not cleaned_chunk:
                continue
            similarity = float(cosine_similarity(self.vectorizer.transform([cleaned_chunk]), reference_vector)[0][0])
            if similarity > best_similarity:
                best_similarity = similarity
                best_chunk = chunk.strip()

        return best_similarity, best_chunk

    def _compute_match_signal(self, cleaned_description: str, skill: str) -> float:
        normalized_skill = preprocess_text(skill)
        padded_description = f" {cleaned_description} "

        if f" {normalized_skill} " in padded_description:
            return 1.0

        for alias in get_skill_aliases(skill, self.config):
            normalized_alias = preprocess_text(alias)
            if normalized_alias and f" {normalized_alias} " in padded_description:
                return 0.8

        return 0.0

    def _combine_signals(self, match_signal: float, similarity: float) -> float:
        return (
            self.config.exact_match_weight * match_signal
            + self.config.semantic_similarity_weight * similarity
        )

    def _calibrate_similarity(self, similarity: float) -> float:
        threshold = self.config.semantic_similarity_strong_threshold
        if threshold <= 0:
            return similarity
        return min(similarity / threshold, 1.0)

    def _to_competency_score(self, hybrid_score: float) -> float:
        return round(hybrid_score * 100, 2)

    def _assign_score_band(self, competency_score: float) -> str:
        selected_band: ScoreBand = self.config.score_bands[0]
        for band in self.config.score_bands:
            if competency_score >= band.minimum_score:
                selected_band = band
        return selected_band.label
