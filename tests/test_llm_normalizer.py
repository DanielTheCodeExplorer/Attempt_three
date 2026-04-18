import json

import pandas as pd

from src.config import PipelineConfig
from src.llm_normalizer import (
    LLM_NORMALIZER_PROMPT,
    SKILL_FOCUSED_REWRITE_PROMPT,
    normalize_biography_dataframe,
    normalize_biography_text,
    rewrite_description_for_skill,
)


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]


class FakeClient:
    class chat:
        class completions:
            @staticmethod
            def create(*args, **kwargs):
                payload = {
                    "standardized_summary": "Delivers Python automation and SQL reporting work.",
                    "matched_skills": [
                        {
                            "skill": "Python",
                            "evidence_phrase": "Python automation",
                            "evidence_sentence": "Built Python automation for reporting.",
                            "evidence_strength": "high",
                        },
                        {
                            "skill": "SQL",
                            "evidence_phrase": "SQL reporting",
                            "evidence_sentence": "Queried data using SQL for reporting.",
                            "evidence_strength": "medium",
                        },
                        {
                            "skill": "Invented Skill",
                            "evidence_phrase": "fake",
                            "evidence_sentence": "fake",
                            "evidence_strength": "high",
                        },
                    ],
                    "clean_skill_evidence_text": "Python automation for reporting. SQL reporting and data querying.",
                }
                return FakeResponse(json.dumps(payload))


class FakeSkillRewriteClient:
    class chat:
        class completions:
            @staticmethod
            def create(*args, **kwargs):
                payload = {
                    "skill_specific_profile": "python scripting automation and reporting",
                }
                return FakeResponse(json.dumps(payload))


class ExplodingClient:
    class chat:
        class completions:
            @staticmethod
            def create(*args, **kwargs):
                raise RuntimeError("boom")


def test_llm_prompt_constant_contains_required_schema():
    assert "Required JSON schema" in LLM_NORMALIZER_PROMPT
    assert "matched_skills" in LLM_NORMALIZER_PROMPT
    assert "clean_skill_evidence_text" in LLM_NORMALIZER_PROMPT


def test_skill_focused_prompt_constant_contains_required_schema():
    assert "Target skill" not in SKILL_FOCUSED_REWRITE_PROMPT
    assert "skill_focused_description" in SKILL_FOCUSED_REWRITE_PROMPT
    assert "evidence_strength" in SKILL_FOCUSED_REWRITE_PROMPT


def test_normalize_biography_text_filters_to_allowed_skills():
    result = normalize_biography_text(
        biography_text="Built Python automation for reporting.",
        description_text="Queried data using SQL for reporting.",
        position="Consultant",
        allowed_skills=["Python", "SQL"],
        config=PipelineConfig(),
        client=FakeClient(),
    )

    assert result["standardized_summary"] == "Delivers Python automation and SQL reporting work."
    assert len(result["matched_skills"]) == 2
    assert [item["skill"] for item in result["matched_skills"]] == ["Python", "SQL"]
    assert result["clean_skill_evidence_text"] == "Python automation for reporting. SQL reporting and data querying."


def test_normalize_biography_dataframe_creates_scoring_description():
    df = pd.DataFrame(
        [
            {
                "talentlinkId": "1001",
                "skills": ["Python", "SQL"],
                "biography": "Built Python automation for reporting.",
                "description": "Queried data using SQL for reporting.",
                "position": "Consultant",
            }
        ]
    )

    normalized = normalize_biography_dataframe(df, config=PipelineConfig(), client=FakeClient())

    assert "clean_skill_evidence_text" in normalized.columns
    assert "description" in normalized.columns
    assert normalized.loc[0, "description"] == normalized.loc[0, "clean_skill_evidence_text"]


def test_normalize_biography_text_fallback_retrieves_biography_skill_evidence():
    config = PipelineConfig(
        skill_references={
            "power bi": "Uses Power BI to build dashboards, visualise data, and communicate performance insights.",
            "data governance": "Supports data governance through data quality controls and standards.",
        },
        skill_aliases={
            "power bi": ("dashboards",),
            "data governance": ("data quality",),
        },
    )

    result = normalize_biography_text(
        biography_text="Built dashboards for executive reporting and improved data quality controls.",
        description_text="Planned office events and budget tracking.",
        position="Consultant",
        allowed_skills=["Power BI", "Data Governance"],
        config=config,
        client=ExplodingClient(),
    )

    assert [item["skill"] for item in result["matched_skills"]] == ["Power BI", "Data Governance"]
    assert "built dashboards for executive reporting" in result["clean_skill_evidence_text"]
    assert "planned office events" not in result["clean_skill_evidence_text"]


def test_normalize_biography_dataframe_fallback_uses_biography_not_job_history():
    config = PipelineConfig(
        skill_references={
            "python": "Uses Python for scripting automation and data analysis.",
        },
        skill_aliases={
            "python": ("python automation",),
        },
    )
    df = pd.DataFrame(
        [
            {
                "talentlinkId": "1001",
                "skills": ["Python"],
                "biography": "Built Python automation for reporting.",
                "description": "Planned office events and catering budgets.",
                "position": "Consultant",
            }
        ]
    )

    normalized = normalize_biography_dataframe(df, config=config, client=ExplodingClient())

    assert normalized.loc[0, "description"] == "built python automation for reporting"
    assert normalized.loc[0, "matched_skills_json"] != "[]"
    assert "Planned office events" not in normalized.loc[0, "clean_skill_evidence_text"]


def test_rewrite_description_for_skill_returns_skill_focused_text():
    payload = rewrite_description_for_skill(
        cleaned_description="planned office events python scripting automation and reporting",
        skill="Python",
        config=PipelineConfig(),
        client=FakeSkillRewriteClient(),
    )

    assert payload["skill_focused_description"] == "python scripting automation and reporting"
    assert payload["evidence_strength"] == "medium"
