import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.config import PipelineConfig
from src.data_loader import parse_skills, standardize_columns
from src.logging_utils import get_logger
from src.skill_references import get_skill_aliases, get_skill_reference
from src.text_preprocessing import normalise_skill_name, preprocess_text, split_text_into_chunks


LOGGER = get_logger(__name__)

SKILL_SPECIFIC_PROFILE_PROMPT = """You are a constrained profile-rewriting component for an employee competency system.

Your task is to write a skill-specific professional summary using only retrieved evidence from the source profile.
The output should mirror the structure and style of the skill reference sentence as closely as the evidence allows.

Rules:
- Use only information explicitly present in the retrieved evidence.
- Do not invent achievements, credentials, projects, tools, or expertise levels.
- Do not add new skills not supported by the evidence.
- Reorganize and compress the evidence into a short professional summary centered on the target skill.
- Use the skill reference as the structural template for the sentence shape, order of ideas, and level of specificity.
- Do not copy unsupported clauses from the skill reference. If part of the reference structure is not supported by the evidence, omit it rather than inventing matching content.
- If the evidence is weak, keep the summary narrow and cautious.
- Prefer one benchmark-style sentence over a loose summary paragraph.
- Return valid JSON only.

Required JSON schema:
{
  "skill_specific_profile": ""
}
"""

RETRIEVAL_MIN_TOKEN_LENGTH = 3


class SupportsChatCompletions(Protocol):
    """Minimal protocol for an injected OpenAI-like client."""

    class chat:  # type: ignore[valid-type]
        class completions:  # type: ignore[valid-type]
            @staticmethod
            def create(*args: Any, **kwargs: Any) -> Any:
                pass


@dataclass(frozen=True)
class RankedChunk:
    """Represents one retrieved chunk and the signals behind its relevance."""

    chunk_text: str
    chunk_index: int
    relevance_score: float
    exact_match_score: float
    alias_match_score: float
    token_overlap_score: float
    reference_overlap_score: float


@dataclass(frozen=True)
class SkillSpecificProfileRecord:
    """Structured output for one skill-specific profile generation request."""

    talentlinkId: str
    target_skill: str
    skill_specific_profile: str
    scoring_text_used: str
    evidence_spans: list[str]
    confidence_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "talentlinkId": self.talentlinkId,
            "target_skill": self.target_skill,
            "skill_specific_profile": self.skill_specific_profile,
            "scoring_text_used": self.scoring_text_used,
            "evidence_spans": self.evidence_spans,
            "confidence_score": self.confidence_score,
        }


def load_data(path: str | Path, config: PipelineConfig | None = None) -> pd.DataFrame:
    """Load a skill-profile input dataset with the target-skill contract."""

    config = config or PipelineConfig()
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    standardized = standardize_columns(df, config).copy()

    required = ["talentlinkId", "description", "skills", "target_skill"]
    missing = [column for column in required if column not in standardized.columns]
    if missing:
        raise ValueError(f"Skill-specific profile dataset is missing required columns: {missing}")

    standardized["talentlinkId"] = standardized["talentlinkId"].astype("string").str.strip()
    standardized["description"] = standardized["description"].fillna("").astype(str).str.strip()
    standardized["skills"] = standardized["skills"].apply(parse_skills)
    standardized["target_skill"] = standardized["target_skill"].fillna("").astype(str).str.strip()

    return standardized.loc[
        standardized["talentlinkId"].ne("")
        & standardized["description"].ne("")
        & standardized["target_skill"].ne("")
    ].reset_index(drop=True)


def chunk_profile_text(description: str) -> list[str]:
    """Split profile text into de-duplicated retrieval chunks."""

    ordered_chunks: list[str] = []
    seen: set[str] = set()
    for chunk in split_text_into_chunks(str(description or "")):
        normalized = chunk.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered_chunks.append(normalized)
    return ordered_chunks


def rank_chunks_for_skill(
    chunks: list[str],
    target_skill: str,
    config: PipelineConfig | None = None,
) -> list[RankedChunk]:
    """Rank profile chunks by transparent lexical relevance to the target skill."""

    config = config or PipelineConfig()
    normalized_skill = normalise_skill_name(target_skill)
    if not normalized_skill:
        return []

    reference_text = get_skill_reference(target_skill, config)
    aliases = get_skill_aliases(target_skill, config)
    reference_tokens = _tokenize_for_retrieval(reference_text)
    skill_tokens = _tokenize_for_retrieval(" ".join([target_skill, reference_text, *aliases]))
    ranked_chunks: list[RankedChunk] = []

    for chunk_index, chunk_text in enumerate(chunks):
        normalized_chunk = preprocess_text(chunk_text)
        if not normalized_chunk:
            continue

        chunk_tokens = _tokenize_for_retrieval(chunk_text)
        padded_chunk = f" {normalized_chunk} "

        exact_match_score = 1.0 if f" {normalized_skill} " in padded_chunk else 0.0
        alias_match_score = 0.0
        for alias in aliases:
            normalized_alias = normalise_skill_name(alias)
            if normalized_alias and f" {normalized_alias} " in padded_chunk:
                alias_match_score = max(alias_match_score, 0.8)

        token_overlap_score = _overlap_ratio(chunk_tokens, skill_tokens)
        reference_overlap_score = _overlap_ratio(chunk_tokens, reference_tokens)

        relevance_score = min(
            1.0,
            round(
                0.55 * max(exact_match_score, alias_match_score)
                + 0.30 * token_overlap_score
                + 0.15 * reference_overlap_score,
                6,
            ),
        )

        ranked_chunks.append(
            RankedChunk(
                chunk_text=chunk_text,
                chunk_index=chunk_index,
                relevance_score=relevance_score,
                exact_match_score=exact_match_score,
                alias_match_score=alias_match_score,
                token_overlap_score=token_overlap_score,
                reference_overlap_score=reference_overlap_score,
            )
        )

    return sorted(
        ranked_chunks,
        key=lambda item: (
            item.relevance_score,
            item.exact_match_score,
            item.alias_match_score,
            item.reference_overlap_score,
            -item.chunk_index,
        ),
        reverse=True,
    )


def generate_skill_specific_profile(
    talentlink_id: str,
    description: str,
    skills: list[str] | str | None,
    target_skill: str,
    config: PipelineConfig | None = None,
    client: SupportsChatCompletions | None = None,
) -> dict[str, Any]:
    """Generate a skill-specific profile by retrieving and rewriting only relevant evidence."""

    config = config or PipelineConfig()
    normalized_skill = str(target_skill or "").strip()
    parsed_skills = parse_skills(skills) if not isinstance(skills, list) else skills
    chunks = chunk_profile_text(description)
    ranked_chunks = rank_chunks_for_skill(chunks, normalized_skill, config)
    selected_chunks = _select_top_chunks(ranked_chunks, parsed_skills, normalized_skill, config)
    evidence_spans = [item.chunk_text for item in sorted(selected_chunks, key=lambda item: item.chunk_index)]
    confidence_score = _compute_confidence_score(selected_chunks, parsed_skills, normalized_skill)

    if not evidence_spans:
        return build_output_record(
            talentlink_id=talentlink_id,
            target_skill=normalized_skill,
            skill_specific_profile="",
            scoring_text_used="",
            evidence_spans=[],
            confidence_score=0.0,
        )

    fallback_profile = _build_evidence_only_profile(evidence_spans)
    profile_text = fallback_profile

    if config.skill_focused_rewrite_enabled and confidence_score >= config.skill_profile_min_llm_confidence:
        rewritten_profile = _rewrite_retrieved_evidence(
            evidence_spans=evidence_spans,
            target_skill=normalized_skill,
            config=config,
            client=client,
        )
        profile_text = _sanitize_generated_profile(rewritten_profile, evidence_spans, fallback_profile)

    scoring_text_used = build_scoring_text_used(
        target_skill=normalized_skill,
        skill_specific_profile=profile_text,
        evidence_spans=evidence_spans,
        confidence_score=confidence_score,
        config=config,
    )

    return build_output_record(
        talentlink_id=talentlink_id,
        target_skill=normalized_skill,
        skill_specific_profile=profile_text,
        scoring_text_used=scoring_text_used,
        evidence_spans=evidence_spans,
        confidence_score=confidence_score,
    )


def build_output_record(
    talentlink_id: str,
    target_skill: str,
    skill_specific_profile: str,
    scoring_text_used: str,
    evidence_spans: list[str],
    confidence_score: float,
) -> dict[str, Any]:
    """Build the final structured output contract for one profile request."""

    record = SkillSpecificProfileRecord(
        talentlinkId=str(talentlink_id or "").strip(),
        target_skill=str(target_skill or "").strip(),
        skill_specific_profile=preprocess_text(skill_specific_profile),
        scoring_text_used=preprocess_text(scoring_text_used),
        evidence_spans=[str(span).strip() for span in evidence_spans if str(span).strip()],
        confidence_score=round(float(confidence_score), 3),
    )
    return record.to_dict()


def generate_skill_specific_profile_dataframe(
    df: pd.DataFrame,
    config: PipelineConfig | None = None,
    client: SupportsChatCompletions | None = None,
) -> pd.DataFrame:
    """Apply skill-specific profile generation row by row over a dataframe."""

    config = config or PipelineConfig()
    records = [
        generate_skill_specific_profile(
            talentlink_id=row.get("talentlinkId", ""),
            description=row.get("description", ""),
            skills=row.get("skills", []),
            target_skill=row.get("target_skill", ""),
            config=config,
            client=client,
        )
        for _, row in df.iterrows()
    ]
    return pd.DataFrame(records)


def build_scoring_text_used(
    target_skill: str,
    skill_specific_profile: str,
    evidence_spans: list[str],
    confidence_score: float,
    config: PipelineConfig,
) -> str:
    """Build a benchmark-aligned scoring text from grounded evidence only."""

    evidence_text = _build_evidence_only_profile(evidence_spans) or preprocess_text(skill_specific_profile)
    if not evidence_text:
        return ""
    if confidence_score < config.skill_profile_reference_alignment_min_confidence:
        return evidence_text

    reference_text = get_skill_reference(target_skill, config)
    evidence_tokens = _tokenize_for_retrieval(evidence_text)
    supported_reference_clauses = [
        preprocess_text(clause)
        for clause in _split_reference_into_clauses(reference_text)
        if _reference_clause_is_supported(clause, evidence_tokens)
    ]
    if not supported_reference_clauses:
        return evidence_text

    parts: list[str] = []
    if (
        _evidence_mentions_target_skill(evidence_text, target_skill, config)
        or confidence_score >= config.skill_profile_reference_skill_prefix_min_confidence
    ):
        parts.append(f"evidence related to {preprocess_text(target_skill)}")
    parts.extend(supported_reference_clauses)
    parts.append(evidence_text)
    return preprocess_text(" ".join(dict.fromkeys(part for part in parts if part)))


def _build_evidence_only_profile(evidence_spans: list[str]) -> str:
    """Create a deterministic, evidence-only fallback profile from retrieved spans."""

    ordered_fragments: list[str] = []
    seen: set[str] = set()
    for span in evidence_spans:
        cleaned_span = preprocess_text(span)
        if not cleaned_span or cleaned_span in seen:
            continue
        seen.add(cleaned_span)
        ordered_fragments.append(cleaned_span)
    return " ".join(ordered_fragments).strip()


def _split_reference_into_clauses(reference_text: str) -> list[str]:
    normalized = str(reference_text or "").replace(" and ", ", ")
    clauses = [clause.strip(" ,.") for clause in normalized.split(",") if clause.strip(" ,.")]
    return clauses or [str(reference_text or "").strip()]


def _compute_confidence_score(
    selected_chunks: list[RankedChunk],
    skills: list[str],
    target_skill: str,
) -> float:
    """Convert retrieval signals into a bounded confidence score."""

    if not selected_chunks:
        return 0.0

    best_score = max(chunk.relevance_score for chunk in selected_chunks)
    average_score = sum(chunk.relevance_score for chunk in selected_chunks) / len(selected_chunks)
    confidence = 0.65 * best_score + 0.25 * average_score + 0.10 * min(len(selected_chunks) / 2, 1.0)

    normalized_skill = normalise_skill_name(target_skill)
    if normalized_skill and normalized_skill in {normalise_skill_name(skill) for skill in skills}:
        confidence += 0.05

    return min(round(confidence, 6), 1.0)


def _overlap_ratio(source_tokens: set[str], target_tokens: set[str]) -> float:
    if not source_tokens or not target_tokens:
        return 0.0
    return round(len(source_tokens & target_tokens) / len(target_tokens), 6)


def _select_top_chunks(
    ranked_chunks: list[RankedChunk],
    skills: list[str],
    target_skill: str,
    config: PipelineConfig,
) -> list[RankedChunk]:
    selected: list[RankedChunk] = []
    for chunk in ranked_chunks:
        if (
            _chunk_mentions_non_target_skill(chunk.chunk_text, skills, target_skill)
            and chunk.exact_match_score == 0
            and chunk.alias_match_score == 0
        ):
            continue
        if (
            chunk.relevance_score >= config.skill_profile_min_chunk_score
            or chunk.exact_match_score > 0
            or chunk.alias_match_score > 0
        ):
            selected.append(chunk)
        if len(selected) >= config.skill_profile_top_k_chunks:
            break
    return selected


def _tokenize_for_retrieval(text: str) -> set[str]:
    return {
        token
        for token in preprocess_text(text).split()
        if len(token) >= RETRIEVAL_MIN_TOKEN_LENGTH and token not in ENGLISH_STOP_WORDS
    }


def _reference_clause_is_supported(clause: str, evidence_tokens: set[str]) -> bool:
    clause_tokens = _tokenize_for_retrieval(clause)
    if not clause_tokens or not evidence_tokens:
        return False
    overlap_count = len(clause_tokens & evidence_tokens)
    return overlap_count >= 1 and overlap_count / len(clause_tokens) >= 0.25


def _evidence_mentions_target_skill(
    evidence_text: str,
    target_skill: str,
    config: PipelineConfig,
) -> bool:
    normalized_evidence = f" {preprocess_text(evidence_text)} "
    normalized_skill = normalise_skill_name(target_skill)
    if normalized_skill and f" {normalized_skill} " in normalized_evidence:
        return True
    for alias in get_skill_aliases(target_skill, config):
        normalized_alias = normalise_skill_name(alias)
        if normalized_alias and f" {normalized_alias} " in normalized_evidence:
            return True
    return False


def _chunk_mentions_non_target_skill(
    chunk_text: str,
    skills: list[str],
    target_skill: str,
) -> bool:
    normalized_chunk = f" {preprocess_text(chunk_text)} "
    normalized_target_skill = normalise_skill_name(target_skill)
    for skill in skills:
        normalized_skill = normalise_skill_name(skill)
        if not normalized_skill or normalized_skill == normalized_target_skill:
            continue
        if f" {normalized_skill} " in normalized_chunk:
            return True
    return False


def _rewrite_retrieved_evidence(
    evidence_spans: list[str],
    target_skill: str,
    config: PipelineConfig,
    client: SupportsChatCompletions | None = None,
) -> str:
    active_client = client or _build_openai_client()
    if active_client is None:
        return ""

    try:
        response = active_client.chat.completions.create(
            model=config.llm_model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=_build_skill_specific_messages(evidence_spans, target_skill, config),
        )
        content = response.choices[0].message.content
        payload = json.loads(content)
        return str(payload.get("skill_specific_profile", "")).strip()
    except Exception as exc:
        LOGGER.warning("skill_specific_profile_rewrite_fallback reason=exception error=%s", exc)
        return ""


def _sanitize_generated_profile(
    generated_profile: str,
    evidence_spans: list[str],
    fallback_profile: str,
) -> str:
    """Reject generated content that is not strongly grounded in retrieved evidence."""

    normalized_generated = str(generated_profile or "").strip()
    if not normalized_generated:
        return fallback_profile

    evidence_chunks = [preprocess_text(span) for span in evidence_spans if preprocess_text(span)]
    evidence_tokens = _tokenize_for_retrieval(" ".join(evidence_spans))
    supported_segments: list[str] = []

    for segment in split_text_into_chunks(normalized_generated) or [normalized_generated]:
        cleaned_segment = preprocess_text(segment)
        if not cleaned_segment:
            continue

        segment_tokens = _tokenize_for_retrieval(segment)
        evidence_precision = _overlap_ratio(segment_tokens, evidence_tokens) if segment_tokens else 0.0
        lexical_support = max(
            (
                _overlap_ratio(segment_tokens, _tokenize_for_retrieval(evidence_chunk))
                for evidence_chunk in evidence_chunks
            ),
            default=0.0,
        )
        evidence_coverage = max(
            (
                _overlap_ratio(_tokenize_for_retrieval(evidence_chunk), segment_tokens)
                for evidence_chunk in evidence_chunks
                if segment_tokens
            ),
            default=0.0,
        )

        if (
            cleaned_segment in evidence_chunks
            or evidence_precision >= 0.85
            or lexical_support >= 0.85
            or evidence_coverage >= 0.6
        ):
            supported_segments.append(cleaned_segment)

    if not supported_segments:
        return fallback_profile

    ordered_supported = list(dict.fromkeys(supported_segments))
    return " ".join(ordered_supported).strip() or fallback_profile


def _build_skill_specific_messages(
    evidence_spans: list[str],
    target_skill: str,
    config: PipelineConfig,
) -> list[dict[str, str]]:
    reference_text = get_skill_reference(target_skill, config)
    evidence_block = json.dumps(evidence_spans, ensure_ascii=True)
    user_message = (
        f"Target skill:\n{target_skill}\n\n"
        f"Skill reference:\n{reference_text}\n\n"
        "Rewrite the evidence so the output follows the reference sentence structure,"
        " but only with claims supported by the evidence.\n\n"
        f"Retrieved evidence spans:\n{evidence_block}\n\n"
        "Return valid JSON only."
    )
    return [
        {"role": "system", "content": SKILL_SPECIFIC_PROFILE_PROMPT},
        {"role": "user", "content": user_message},
    ]


def _build_openai_client() -> SupportsChatCompletions | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        LOGGER.warning("skill_specific_profile_missing_dependency dependency=openai")
        return None

    if config_base_url := os.getenv("OPENAI_BASE_URL"):
        return OpenAI(api_key=api_key, base_url=config_base_url)
    return OpenAI(api_key=api_key)
