from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.config import PipelineConfig
from src.logging_utils import get_logger
from src.skill_taxonomy import SkillAliasDefinition, SkillDefinition, SkillTaxonomy
from src.text_preprocessing import preprocess_text, split_text_into_chunks


LOGGER = get_logger(__name__)


class SupportsEmbeddings(Protocol):
    """Minimal embedding interface so the pipeline can inject a fake model in tests."""

    def encode(self, texts: list[str]) -> np.ndarray:
        pass


EMBEDDING_MODEL_CACHE: dict[str, SupportsEmbeddings | None] = {}


@dataclass(frozen=True)
class MatchedAliasEvidence:
    """Evidence for one alias hit in the input text."""

    alias_id: str
    alias_text: str
    skill_id: str
    canonical_name: str
    alias_type: str
    weight: float
    evidence_snippet: str


@dataclass(frozen=True)
class ExtractedSkill:
    """One canonical skill detected in the text."""

    skill_id: str
    canonical_name: str
    source: str
    taxonomy_score: float
    matched_aliases: tuple[str, ...]
    evidence_snippets: tuple[str, ...]


def preprocess_for_matching(text: str) -> tuple[str, list[str], set[str]]:
    """Normalize text and precompute token and n-gram sets for alias matching."""

    normalized = preprocess_text(text)
    tokens = [token for token in normalized.split() if token]
    max_n = min(5, len(tokens)) if tokens else 0
    ngrams = set()
    for n in range(1, max_n + 1):
        for index in range(len(tokens) - n + 1):
            ngrams.add(" ".join(tokens[index : index + n]))
    return normalized, tokens, ngrams


def extract_skills(
    text: str,
    taxonomy: SkillTaxonomy,
    config: PipelineConfig | None = None,
    embedding_model: SupportsEmbeddings | None = None,
) -> dict[str, object]:
    """Extract canonical skills from text using aliases first and embeddings as fallback."""

    config = config or PipelineConfig()
    normalized_text, _, ngrams = preprocess_for_matching(text)
    matched_aliases = match_aliases(text, normalized_text, ngrams, taxonomy)

    if matched_aliases:
        extracted_skills = map_aliases_to_skills(matched_aliases, taxonomy)
        return {
            "matched_skills": [skill.__dict__ for skill in extracted_skills],
            "matched_aliases": [alias.__dict__ for alias in matched_aliases],
            "evidence_snippets": sorted(
                {
                    snippet
                    for skill in extracted_skills
                    for snippet in skill.evidence_snippets
                    if snippet
                }
            ),
        }

    embedded_skill = fallback_semantic_match(
        text=text,
        taxonomy=taxonomy,
        threshold=config.embedding_similarity_threshold,
        embedding_model=embedding_model,
        model_name=config.sentence_transformer_model_name,
    )
    if embedded_skill is None:
        return {"matched_skills": [], "matched_aliases": [], "evidence_snippets": []}

    return {
        "matched_skills": [embedded_skill.__dict__],
        "matched_aliases": [],
        "evidence_snippets": list(embedded_skill.evidence_snippets),
    }


def match_aliases(
    original_text: str,
    normalized_text: str,
    ngrams: set[str],
    taxonomy: SkillTaxonomy,
) -> list[MatchedAliasEvidence]:
    """Find explicit canonical-skill or alias matches inside the text."""

    matches: list[MatchedAliasEvidence] = []
    for skill in taxonomy.skills:
        canonical_alias = preprocess_text(skill.canonical_name)
        if canonical_alias and canonical_alias in ngrams:
            matches.append(
                MatchedAliasEvidence(
                    alias_id=f"CANONICAL_{skill.skill_id}",
                    alias_text=skill.canonical_name,
                    skill_id=skill.skill_id,
                    canonical_name=skill.canonical_name,
                    alias_type="exact",
                    weight=1.0,
                    evidence_snippet=find_evidence_snippet(original_text, canonical_alias),
                )
            )

    for alias in taxonomy.aliases:
        normalized_alias = preprocess_text(alias.alias_text)
        if not normalized_alias:
            continue
        if normalized_alias not in ngrams and normalized_alias not in normalized_text:
            continue
        skill = taxonomy.skill_by_id[alias.skill_id]
        matches.append(
            MatchedAliasEvidence(
                alias_id=alias.alias_id,
                alias_text=alias.alias_text,
                skill_id=alias.skill_id,
                canonical_name=skill.canonical_name,
                alias_type=alias.alias_type,
                weight=alias.weight,
                evidence_snippet=find_evidence_snippet(original_text, normalized_alias),
            )
        )
    return matches


def map_aliases_to_skills(
    matched_aliases: list[MatchedAliasEvidence],
    taxonomy: SkillTaxonomy,
) -> list[ExtractedSkill]:
    """Collapse multiple alias hits into one canonical-skill extraction result."""

    grouped: dict[str, list[MatchedAliasEvidence]] = {}
    for match in matched_aliases:
        grouped.setdefault(match.skill_id, []).append(match)

    extracted: list[ExtractedSkill] = []
    for skill_id, items in grouped.items():
        skill = taxonomy.skill_by_id[skill_id]
        best_weight = max(item.weight for item in items)
        evidence = tuple(dict.fromkeys(item.evidence_snippet for item in items if item.evidence_snippet))
        aliases = tuple(dict.fromkeys(item.alias_text for item in items if item.alias_text))
        extracted.append(
            ExtractedSkill(
                skill_id=skill_id,
                canonical_name=skill.canonical_name,
                source="alias",
                taxonomy_score=round(best_weight, 6),
                matched_aliases=aliases,
                evidence_snippets=evidence,
            )
        )
    return sorted(extracted, key=lambda item: item.taxonomy_score, reverse=True)


def fallback_semantic_match(
    text: str,
    taxonomy: SkillTaxonomy,
    threshold: float,
    embedding_model: SupportsEmbeddings | None = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> ExtractedSkill | None:
    """Use sentence-transformer style embeddings when lexical alias matching fails."""

    model = embedding_model or build_default_embedding_model(model_name)
    if model is None:
        return None

    candidate_entries = build_embedding_candidates(taxonomy)
    query_embedding = np.asarray(model.encode([text]), dtype=float)
    candidate_embeddings = np.asarray(model.encode([item["text"] for item in candidate_entries]), dtype=float)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    if candidate_embeddings.ndim == 1:
        candidate_embeddings = candidate_embeddings.reshape(1, -1)

    similarities = cosine_similarity_matrix(query_embedding, candidate_embeddings)[0]
    best_by_skill: dict[str, tuple[float, str]] = {}
    for similarity, candidate in zip(similarities, candidate_entries):
        skill_id = candidate["skill_id"]
        current = best_by_skill.get(skill_id)
        if current is None or similarity > current[0]:
            best_by_skill[skill_id] = (float(similarity), candidate["text"])

    if not best_by_skill:
        return None

    best_skill_id, (best_similarity, candidate_text) = max(best_by_skill.items(), key=lambda item: item[1][0])
    if best_similarity < threshold:
        return None

    skill = taxonomy.skill_by_id[best_skill_id]
    evidence_snippet = find_evidence_snippet(text, candidate_text)
    return ExtractedSkill(
        skill_id=best_skill_id,
        canonical_name=skill.canonical_name,
        source="embedding",
        taxonomy_score=round(best_similarity, 6),
        matched_aliases=tuple(),
        evidence_snippets=(evidence_snippet,) if evidence_snippet else tuple(),
    )


def build_embedding_candidates(taxonomy: SkillTaxonomy) -> list[dict[str, str]]:
    """Create canonical and alias candidate texts for embedding fallback."""

    entries: list[dict[str, str]] = []
    for skill in taxonomy.skills:
        entries.append(
            {
                "skill_id": skill.skill_id,
                "text": f"{skill.canonical_name}. {skill.description}",
            }
        )
    for alias in taxonomy.aliases:
        entries.append({"skill_id": alias.skill_id, "text": alias.alias_text})
    return entries


def build_default_embedding_model(model_name: str) -> SupportsEmbeddings | None:
    """Instantiate the default sentence-transformer lazily so import remains optional."""

    if model_name in EMBEDDING_MODEL_CACHE:
        return EMBEDDING_MODEL_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        EMBEDDING_MODEL_CACHE[model_name] = None
        return None
    try:
        model = SentenceTransformer(model_name)
        EMBEDDING_MODEL_CACHE[model_name] = model
        return model
    except Exception as exc:
        LOGGER.warning("embedding_model_unavailable model=%s error=%s", model_name, exc)
        EMBEDDING_MODEL_CACHE[model_name] = None
        return None


def find_evidence_snippet(text: str, query_text: str) -> str:
    """Select the sentence or chunk that best supports a matched alias or semantic candidate."""

    chunks = split_text_into_chunks(text)
    if not chunks:
        return ""

    normalized_query = preprocess_text(query_text)
    query_tokens = set(normalized_query.split())
    best_chunk = chunks[0]
    best_score = -1
    for chunk in chunks:
        normalized_chunk = preprocess_text(chunk)
        if normalized_query and normalized_query in normalized_chunk:
            return chunk
        chunk_tokens = set(normalized_chunk.split())
        score = len(chunk_tokens & query_tokens)
        if score > best_score:
            best_score = score
            best_chunk = chunk
    return best_chunk


def cosine_similarity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two dense embedding matrices."""

    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    safe_left = np.divide(left, np.clip(left_norm, a_min=1e-12, a_max=None))
    safe_right = np.divide(right, np.clip(right_norm, a_min=1e-12, a_max=None))
    return safe_left @ safe_right.T
