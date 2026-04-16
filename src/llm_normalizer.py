import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from src.config import PipelineConfig
from src.data_loader import parse_skills
from src.logging_utils import get_logger
from src.text_preprocessing import normalise_skill_name


LOGGER = get_logger(__name__)

LLM_NORMALIZER_PROMPT = """You are an information extraction component for an employee competency scoring system.

Your task is to read employee biography and description text and convert them into structured skill evidence for downstream scoring.

Rules:
- Use only information explicitly present in the source text.
- Do not invent or assume skills.
- If an allowed skill list is provided, only return skills from that list.
- Prefer precision over recall.
- Return valid JSON only.
- If evidence is weak, return fewer skills rather than guessing.

Required JSON schema:
{
  "standardized_summary": "",
  "matched_skills": [
    {
      "skill": "",
      "evidence_phrase": "",
      "evidence_sentence": "",
      "evidence_strength": "high|medium|low"
    }
  ],
  "clean_skill_evidence_text": ""
}

Generation requirements:
- `standardized_summary` should rewrite the source text into short, standardised professional wording.
- `matched_skills` should contain only skills explicitly supported by the source text.
- `clean_skill_evidence_text` should be a compact paragraph of standardised skill-evidence statements suitable for TF-IDF or similarity scoring.
- If biography and description overlap, deduplicate the evidence.
- Preserve technical terms exactly where possible."""

SKILL_FOCUSED_REWRITE_PROMPT = """You are a text normalization component for an employee competency scoring system.

Your task is to rewrite a cleaned employee description so it focuses only on evidence relevant to one target skill.

Rules:
- Use only information explicitly present in the cleaned description.
- Do not invent experience, tools, tasks, or outcomes.
- Keep the output in the same compact cleaned-description style as the input.
- Remove content unrelated to the target skill where possible.
- If the description contains weak or no explicit evidence for the target skill, return the original cleaned description unchanged.
- Return valid JSON only.

Required JSON schema:
{
  "skill_focused_description": "",
  "evidence_strength": "high|medium|low"
}
"""


class SupportsChatCompletions(Protocol):
    """Minimal protocol for an injected OpenAI-like client."""

    class chat:  # type: ignore[valid-type]
        class completions:  # type: ignore[valid-type]
            @staticmethod
            def create(*args: Any, **kwargs: Any) -> Any:
                pass


@dataclass(frozen=True)
class MatchedSkillEvidence:
    """Structured evidence extracted for one matched skill."""

    skill: str
    evidence_phrase: str
    evidence_sentence: str
    evidence_strength: str


@dataclass(frozen=True)
class NormalizedBiographyResult:
    """Validated normalized biography payload used downstream."""

    standardized_summary: str
    matched_skills: list[MatchedSkillEvidence]
    clean_skill_evidence_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "standardized_summary": self.standardized_summary,
            "matched_skills": [skill.__dict__ for skill in self.matched_skills],
            "clean_skill_evidence_text": self.clean_skill_evidence_text,
        }


@dataclass(frozen=True)
class SkillFocusedRewriteResult:
    """Validated skill-focused rewrite payload used during scoring."""

    skill_focused_description: str
    evidence_strength: str

    def to_dict(self) -> dict[str, str]:
        return {
            "skill_focused_description": self.skill_focused_description,
            "evidence_strength": self.evidence_strength,
        }


def normalize_biography_text(
    biography_text: str,
    description_text: str,
    position: str | None = None,
    allowed_skills: list[str] | None = None,
    config: PipelineConfig | None = None,
    client: SupportsChatCompletions | None = None,
) -> dict[str, Any]:
    """Normalize biography and description text into structured skill evidence using an LLM."""

    config = config or PipelineConfig()
    biography = str(biography_text or "").strip()
    description = str(description_text or "").strip()
    role = str(position or "").strip()
    if not biography and not description:
        return empty_normalization_result().to_dict()
    source_text = build_source_text(biography, description, role)

    try:
        active_client = client or build_openai_client()
        if active_client is None:
            LOGGER.warning("llm_normalizer_fallback reason=no_client")
            return conservative_fallback_result(source_text, allowed_skills).to_dict()

        response = active_client.chat.completions.create(
            model=config.llm_model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=build_messages(biography, description, role, allowed_skills),
        )
        content = response.choices[0].message.content
        payload = json.loads(content)
        return validate_normalized_payload(payload, source_text, allowed_skills).to_dict()
    except Exception as exc:
        LOGGER.warning("llm_normalizer_fallback reason=exception error=%s", exc)
        return conservative_fallback_result(source_text, allowed_skills).to_dict()


def rewrite_description_for_skill(
    cleaned_description: str,
    skill: str,
    config: PipelineConfig | None = None,
    client: SupportsChatCompletions | None = None,
) -> dict[str, str]:
    """Rewrite a cleaned description so it focuses on one target skill."""

    config = config or PipelineConfig()
    normalized_description = str(cleaned_description or "").strip()
    normalized_skill = str(skill or "").strip()
    if not normalized_description or not normalized_skill:
        return SkillFocusedRewriteResult(
            skill_focused_description=normalized_description,
            evidence_strength="low",
        ).to_dict()

    try:
        active_client = client or build_openai_client()
        if active_client is None:
            return fallback_skill_focused_rewrite(normalized_description, normalized_skill).to_dict()

        response = active_client.chat.completions.create(
            model=config.llm_model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=build_skill_focused_messages(normalized_description, normalized_skill),
        )
        content = response.choices[0].message.content
        payload = json.loads(content)
        return validate_skill_focused_payload(payload, normalized_description).to_dict()
    except Exception as exc:
        LOGGER.warning("skill_focused_rewrite_fallback reason=exception error=%s", exc)
        return fallback_skill_focused_rewrite(normalized_description, normalized_skill).to_dict()


def normalize_biography_dataframe(
    df: pd.DataFrame,
    config: PipelineConfig | None = None,
    client: SupportsChatCompletions | None = None,
) -> pd.DataFrame:
    """Apply biography normalization row by row and append structured output columns."""

    config = config or PipelineConfig()
    working = df.copy()
    normalized_payloads = [
        normalize_biography_text(
            biography_text=row.get("biography", ""),
            description_text=row.get("description", ""),
            position=row.get("position", ""),
            allowed_skills=parse_skills(row.get("skills", None)),
            config=config,
            client=client,
        )
        for _, row in working.iterrows()
    ]

    working["standardized_summary"] = [payload["standardized_summary"] for payload in normalized_payloads]
    working["matched_skills_json"] = [json.dumps(payload["matched_skills"], ensure_ascii=True) for payload in normalized_payloads]
    working["clean_skill_evidence_text"] = [payload["clean_skill_evidence_text"] for payload in normalized_payloads]
    working["description"] = working["clean_skill_evidence_text"].where(
        working["clean_skill_evidence_text"].astype(str).str.strip().ne(""),
        working["description"].where(
            working["description"].astype(str).str.strip().ne(""),
            working["biography"],
        ),
    )
    return working


def build_messages(
    biography_text: str,
    description_text: str,
    position: str | None,
    allowed_skills: list[str] | None,
) -> list[dict[str, str]]:
    """Build the exact LLM prompt messages for normalization."""

    skill_clause = json.dumps(allowed_skills or [], ensure_ascii=True)
    user_message = (
        f"Biography text:\n{biography_text}\n\n"
        f"Description text:\n{description_text}\n\n"
        f"Position:\n{position or ''}\n\n"
        f"Allowed skills:\n{skill_clause}\n\n"
        "Return valid JSON only."
    )
    return [
        {"role": "system", "content": LLM_NORMALIZER_PROMPT},
        {"role": "user", "content": user_message},
    ]


def build_skill_focused_messages(
    cleaned_description: str,
    skill: str,
) -> list[dict[str, str]]:
    """Build the exact LLM prompt messages for skill-focused description rewriting."""

    user_message = (
        f"Target skill:\n{skill}\n\n"
        f"Cleaned description:\n{cleaned_description}\n\n"
        "Return valid JSON only."
    )
    return [
        {"role": "system", "content": SKILL_FOCUSED_REWRITE_PROMPT},
        {"role": "user", "content": user_message},
    ]


def build_openai_client() -> SupportsChatCompletions | None:
    """Create an OpenAI client if the SDK and API key are available."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        LOGGER.warning("llm_normalizer_missing_dependency dependency=openai")
        return None

    if config_base_url := os.getenv("OPENAI_BASE_URL"):
        return OpenAI(api_key=api_key, base_url=config_base_url)
    return OpenAI(api_key=api_key)


def build_source_text(biography: str, description: str, position: str) -> str:
    """Combine biography, description, and position into one source block."""

    parts = []
    if position:
        parts.append(f"Position: {position}")
    if biography:
        parts.append(f"Biography: {biography}")
    if description:
        parts.append(f"Description: {description}")
    return "\n".join(parts).strip()


def validate_skill_focused_payload(
    payload: dict[str, Any],
    cleaned_description: str,
) -> SkillFocusedRewriteResult:
    """Validate and sanitize the LLM JSON response for a skill-focused rewrite."""

    rewritten_description = str(payload.get("skill_focused_description", "")).strip() or cleaned_description
    evidence_strength = str(payload.get("evidence_strength", "low")).strip().lower()
    if evidence_strength not in {"high", "medium", "low"}:
        evidence_strength = "low"
    return SkillFocusedRewriteResult(
        skill_focused_description=rewritten_description,
        evidence_strength=evidence_strength,
    )


def validate_normalized_payload(
    payload: dict[str, Any],
    biography: str,
    allowed_skills: list[str] | None,
) -> NormalizedBiographyResult:
    """Validate and sanitize the LLM JSON response."""

    allowed_lookup = {normalise_skill_name(skill): skill for skill in (allowed_skills or [])}
    matched_skills: list[MatchedSkillEvidence] = []

    for item in payload.get("matched_skills", []):
        raw_skill = str(item.get("skill", "")).strip()
        normalized_skill = normalise_skill_name(raw_skill)
        if not raw_skill:
            continue
        if allowed_lookup and normalized_skill not in allowed_lookup:
            continue

        evidence_sentence = str(item.get("evidence_sentence", "")).strip()
        evidence_phrase = str(item.get("evidence_phrase", "")).strip()
        if not evidence_sentence and not evidence_phrase:
            continue

        strength = str(item.get("evidence_strength", "low")).strip().lower()
        if strength not in {"high", "medium", "low"}:
            strength = "low"

        matched_skills.append(
            MatchedSkillEvidence(
                skill=allowed_lookup.get(normalized_skill, raw_skill),
                evidence_phrase=evidence_phrase,
                evidence_sentence=evidence_sentence,
                evidence_strength=strength,
            )
        )

    standardized_summary = str(payload.get("standardized_summary", "")).strip() or biography
    clean_skill_evidence_text = str(payload.get("clean_skill_evidence_text", "")).strip()
    if not clean_skill_evidence_text:
        clean_skill_evidence_text = standardized_summary

    return NormalizedBiographyResult(
        standardized_summary=standardized_summary,
        matched_skills=matched_skills,
        clean_skill_evidence_text=clean_skill_evidence_text,
    )


def conservative_fallback_result(
    biography: str,
    allowed_skills: list[str] | None,
) -> NormalizedBiographyResult:
    """Return a conservative fallback result when the LLM is unavailable or fails."""

    normalized_biography = biography.strip()
    matched_skills: list[MatchedSkillEvidence] = []

    if allowed_skills:
        lowered_text = f" {normalise_skill_name(normalized_biography)} "
        for skill in allowed_skills:
            normalized_skill = normalise_skill_name(skill)
            if normalized_skill and f" {normalized_skill} " in lowered_text:
                matched_skills.append(
                    MatchedSkillEvidence(
                        skill=skill,
                        evidence_phrase=skill,
                        evidence_sentence=biography,
                        evidence_strength="high",
                    )
                )

    clean_text = " ".join(
        f"{item.skill}: {item.evidence_phrase}."
        for item in matched_skills
    ).strip() or normalized_biography

    return NormalizedBiographyResult(
        standardized_summary=normalized_biography,
        matched_skills=matched_skills,
        clean_skill_evidence_text=clean_text,
    )


def fallback_skill_focused_rewrite(
    cleaned_description: str,
    skill: str,
) -> SkillFocusedRewriteResult:
    """Return the original cleaned description when no safe skill-focused rewrite is available."""

    return SkillFocusedRewriteResult(
        skill_focused_description=cleaned_description.strip(),
        evidence_strength="low",
    )


def empty_normalization_result() -> NormalizedBiographyResult:
    """Return an empty normalization payload."""

    return NormalizedBiographyResult(
        standardized_summary="",
        matched_skills=[],
        clean_skill_evidence_text="",
    )
