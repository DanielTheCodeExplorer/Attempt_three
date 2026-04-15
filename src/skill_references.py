from src.config import PipelineConfig
from src.text_preprocessing import normalise_skill_name


def get_skill_reference(skill: str, config: PipelineConfig) -> str:
    """Return the configured reference text for a skill, or a controlled fallback."""

    normalized_skill = normalise_skill_name(skill)
    if normalized_skill in config.skill_references:
        return config.skill_references[normalized_skill]

    return config.default_skill_reference_template.format(skill=skill)


def get_skill_aliases(skill: str, config: PipelineConfig) -> tuple[str, ...]:
    """Return configured aliases or evidence phrases for a skill."""

    normalized_skill = normalise_skill_name(skill)
    return config.skill_aliases.get(normalized_skill, ())
