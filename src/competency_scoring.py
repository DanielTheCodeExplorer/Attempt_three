from dataclasses import dataclass
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import PipelineConfig, ScoreBand
from src.logging_utils import get_logger
from src.skill_extraction import SupportsEmbeddings, extract_skills
from src.skill_taxonomy import SkillTaxonomy, build_default_skill_taxonomy, export_default_taxonomy_csvs
from src.text_preprocessing import preprocess_text


LOGGER = get_logger(__name__)


@dataclass
class ScoreRecord:
    """Represents one employee-skill taxonomy scoring row."""

    talentlinkId: str
    skill_id: str
    skill: str
    category: str
    cleaned_description: str
    source: str
    matched_aliases: str
    evidence_snippets: str
    taxonomy_score: float
    tfidf_score: float
    similarity_score: float
    graph_boost_score: float
    final_score: float
    competency_score: float
    employee_average_competency_score: float
    score_band: str


class CompetencyScorer:
    """Taxonomy-driven scorer that combines alias matching, embeddings, TF-IDF, and graph boosts."""

    def __init__(
        self,
        config: PipelineConfig,
        taxonomy: SkillTaxonomy | None = None,
        embedding_model: SupportsEmbeddings | None = None,
    ) -> None:
        self.config = config
        self.taxonomy = taxonomy or build_default_skill_taxonomy()
        self.embedding_model = embedding_model
        self._score_cache: dict[str, dict[str, dict[str, object]]] = {}

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate one audit row per employee-canonical-skill pair."""

        self._validate_input_frame(df)
        working = df.copy()
        working["base_description"] = working["description"].fillna("").astype(str).str.strip()
        working["cleaned_description"] = working["base_description"].apply(preprocess_text)

        records: list[ScoreRecord] = []
        for _, row in working.reset_index(drop=True).iterrows():
            score_lookup = self._score_all_skills(row["base_description"])
            for skill in self.taxonomy.skills:
                skill_score = score_lookup[skill.skill_id]
                record = self._build_score_record(
                    talentlink_id=row["talentlinkId"],
                    skill_id=skill.skill_id,
                    skill=skill.canonical_name,
                    category=skill.category,
                    cleaned_description=row["cleaned_description"],
                    score_payload=skill_score,
                )
                records.append(record)

        records = self._assign_employee_averages(records)
        LOGGER.info("score_dataframe_complete employees=%s scores=%s", len(working), len(records))
        return pd.DataFrame([record.__dict__ for record in records])

    def _validate_input_frame(self, df: pd.DataFrame) -> None:
        required = {"talentlinkId", "description"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Scoring dataframe is missing required columns: {sorted(missing)}")

    def score_text(self, text: str, min_score: float | None = None) -> dict[str, list[dict[str, object]]]:
        """Return a structured skill list for one text input."""

        min_score = self.config.minimum_output_skill_score if min_score is None else min_score
        scored = self._score_all_skills(text)
        items: list[dict[str, object]] = []
        for skill in self.taxonomy.skills:
            payload = scored[skill.skill_id]
            final_score = float(payload["final_score"])
            if final_score <= 0 or final_score < min_score:
                continue
            items.append(
                {
                    "skill": skill.canonical_name,
                    "score": round(final_score, 6),
                    "evidence": list(payload["evidence_snippets"]),
                    "source": payload["source"],
                    "matched_aliases": list(payload["matched_aliases"]),
                    "tfidf_score": round(float(payload["tfidf_score"]), 6),
                    "taxonomy_score": round(float(payload["taxonomy_score"]), 6),
                    "graph_boost_score": round(float(payload["graph_boost_score"]), 6),
                }
            )
        items.sort(key=lambda item: item["score"], reverse=True)
        return {"skills": items}

    def export_taxonomy_csvs(self, output_dir: str | None = None) -> dict[str, object]:
        """Export the default taxonomy datasets to CSV files."""

        export_dir = output_dir or str(self.config.taxonomy_output_dir)
        return export_default_taxonomy_csvs(export_dir)

    def _score_all_skills(self, description: str) -> dict[str, dict[str, object]]:
        cache_key = preprocess_text(description)
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        extraction = extract_skills(
            text=description,
            taxonomy=self.taxonomy,
            config=self.config,
            embedding_model=self.embedding_model,
        )
        tfidf_scores = compute_tfidf_scores(description, self.taxonomy.skill_profiles(), self.config)
        direct_scores = self._build_direct_scores(extraction, tfidf_scores)
        graph_boosts = self._compute_graph_boosts(direct_scores)

        combined: dict[str, dict[str, object]] = {}
        for skill in self.taxonomy.skills:
            direct_payload = direct_scores.get(
                skill.skill_id,
                {
                    "source": "",
                    "matched_aliases": tuple(),
                    "evidence_snippets": tuple(),
                    "taxonomy_score": 0.0,
                    "tfidf_score": float(tfidf_scores.get(skill.canonical_name, 0.0)),
                    "base_score": 0.0,
                },
            )
            graph_boost = graph_boosts.get(skill.skill_id, 0.0)
            final_score = min(1.0, float(direct_payload["base_score"]) + graph_boost)
            combined[skill.skill_id] = {
                "source": direct_payload["source"] or ("graph" if graph_boost > 0 else ""),
                "matched_aliases": tuple(direct_payload["matched_aliases"]),
                "evidence_snippets": tuple(direct_payload["evidence_snippets"]),
                "taxonomy_score": round(float(direct_payload["taxonomy_score"]), 6),
                "tfidf_score": round(float(direct_payload["tfidf_score"]), 6),
                "graph_boost_score": round(float(graph_boost), 6),
                "final_score": round(final_score, 6),
            }

        self._score_cache[cache_key] = combined
        return combined

    def _build_direct_scores(
        self,
        extraction: dict[str, object],
        tfidf_scores: dict[str, float],
    ) -> dict[str, dict[str, object]]:
        direct_scores: dict[str, dict[str, object]] = {}
        skill_by_name = self.taxonomy.skill_by_name
        for item in extraction.get("matched_skills", []):
            skill_name = str(item.get("canonical_name", item.get("skill", ""))).strip()
            skill = skill_by_name.get(skill_name.lower())
            if skill is None:
                continue
            source = str(item.get("source", "")).strip().lower()
            taxonomy_score = float(item.get("taxonomy_score", 0.0))
            tfidf_score = float(tfidf_scores.get(skill.canonical_name, 0.0))
            if source == "embedding":
                base_score = (
                    self.config.tfidf_weight_with_embedding * tfidf_score
                    + self.config.embedding_weight_in_final * taxonomy_score
                )
            else:
                base_score = (
                    self.config.tfidf_weight_with_alias * tfidf_score
                    + self.config.alias_weight_in_final * taxonomy_score
                )
            direct_scores[skill.skill_id] = {
                "source": source or "alias",
                "matched_aliases": tuple(item.get("matched_aliases", tuple())),
                "evidence_snippets": tuple(item.get("evidence_snippets", tuple())),
                "taxonomy_score": taxonomy_score,
                "tfidf_score": tfidf_score,
                "base_score": min(1.0, base_score),
            }
        return direct_scores

    def _compute_graph_boosts(self, direct_scores: dict[str, dict[str, object]]) -> dict[str, float]:
        boosts: dict[str, float] = {}
        outgoing = self.taxonomy.outgoing_edges
        for source_skill_id, payload in direct_scores.items():
            source_score = float(payload["base_score"])
            if source_score <= 0:
                continue
            for edge in outgoing.get(source_skill_id, tuple()):
                boost = source_score * edge.weight * self.config.graph_boost_factor
                boosts[edge.target_skill_id] = boosts.get(edge.target_skill_id, 0.0) + boost
        return {skill_id: min(1.0, boost) for skill_id, boost in boosts.items()}

    def _build_score_record(
        self,
        talentlink_id: str,
        skill_id: str,
        skill: str,
        category: str,
        cleaned_description: str,
        score_payload: dict[str, object],
    ) -> ScoreRecord:
        final_score = float(score_payload["final_score"])
        competency_score = round(final_score * 100, 2)

        return ScoreRecord(
            talentlinkId=talentlink_id,
            skill_id=skill_id,
            skill=skill,
            category=category,
            cleaned_description=cleaned_description,
            source=str(score_payload["source"]),
            matched_aliases=json.dumps(list(score_payload["matched_aliases"]), ensure_ascii=True),
            evidence_snippets=json.dumps(list(score_payload["evidence_snippets"]), ensure_ascii=True),
            taxonomy_score=round(float(score_payload["taxonomy_score"]), 6),
            tfidf_score=round(float(score_payload["tfidf_score"]), 6),
            similarity_score=round(float(score_payload["tfidf_score"]), 6),
            graph_boost_score=round(float(score_payload["graph_boost_score"]), 6),
            final_score=round(final_score, 6),
            competency_score=competency_score,
            employee_average_competency_score=0.0,
            score_band=self._assign_score_band(competency_score),
        )

    def _assign_employee_averages(self, records: list[ScoreRecord]) -> list[ScoreRecord]:
        grouped: dict[str, list[ScoreRecord]] = {}
        for record in records:
            grouped.setdefault(record.talentlinkId, []).append(record)

        updated: list[ScoreRecord] = []
        for talentlink_id, group in grouped.items():
            average_score = round(sum(record.competency_score for record in group) / len(group), 2) if group else 0.0
            for record in group:
                updated.append(
                    ScoreRecord(
                        talentlinkId=record.talentlinkId,
                        skill_id=record.skill_id,
                        skill=record.skill,
                        category=record.category,
                        cleaned_description=record.cleaned_description,
                        source=record.source,
                        matched_aliases=record.matched_aliases,
                        evidence_snippets=record.evidence_snippets,
                        taxonomy_score=record.taxonomy_score,
                        tfidf_score=record.tfidf_score,
                        similarity_score=record.similarity_score,
                        graph_boost_score=record.graph_boost_score,
                        final_score=record.final_score,
                        competency_score=record.competency_score,
                        employee_average_competency_score=average_score,
                        score_band=record.score_band,
                    )
                )
        return updated

    def _assign_score_band(self, competency_score: float) -> str:
        selected_band: ScoreBand = self.config.score_bands[0]
        for band in self.config.score_bands:
            if competency_score >= band.minimum_score:
                selected_band = band
        return selected_band.label


def compute_tfidf_scores(
    text: str,
    skill_profiles: dict[str, str],
    config: PipelineConfig | None = None,
) -> dict[str, float]:
    """Compute cosine similarity between the input text and each skill profile."""

    config = config or PipelineConfig()
    normalized_text = preprocess_text(text)
    profiles = {name: preprocess_text(profile) for name, profile in skill_profiles.items()}
    corpus = [normalized_text, *profiles.values()]
    vectorizer = TfidfVectorizer(
        max_features=config.tfidf_max_features,
        ngram_range=config.tfidf_ngram_range,
        stop_words=config.tfidf_stop_words,
        sublinear_tf=config.tfidf_sublinear_tf,
    )
    matrix = vectorizer.fit_transform(corpus)
    text_vector = matrix[0]
    profile_vectors = matrix[1:]
    similarities = cosine_similarity(text_vector, profile_vectors)[0]
    return {
        skill_name: round(float(score), 6)
        for skill_name, score in zip(profiles.keys(), similarities)
    }
