from src.config import PipelineConfig
from src.text_preprocessing import normalise_skill_name


def get_skill_reference(skill: str, config: PipelineConfig) -> str:
    """Return the configured reference text for a skill, or a controlled fallback."""

    normalized_skill = normalise_skill_name(skill)
    if normalized_skill in config.skill_references:
        return config.skill_references[normalized_skill]

    return config.default_skill_reference_template.format(skill=skill)
