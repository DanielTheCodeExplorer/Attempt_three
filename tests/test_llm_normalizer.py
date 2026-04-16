import json

import pandas as pd

from src.config import PipelineConfig
from src.llm_normalizer import LLM_NORMALIZER_PROMPT, normalize_biography_dataframe, normalize_biography_text


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


def test_llm_prompt_constant_contains_required_schema():
    assert "Required JSON schema" in LLM_NORMALIZER_PROMPT
    assert "matched_skills" in LLM_NORMALIZER_PROMPT
    assert "clean_skill_evidence_text" in LLM_NORMALIZER_PROMPT


def test_normalize_biography_text_filters_to_allowed_skills():
    result = normalize_biography_text(
        raw_biography_text="Built Python automation for reporting and queried data using SQL.",
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
                "biography": "Built Python automation for reporting and queried data using SQL.",
            }
        ]
    )

    normalized = normalize_biography_dataframe(df, config=PipelineConfig(), client=FakeClient())

    assert "clean_skill_evidence_text" in normalized.columns
    assert "description" in normalized.columns
    assert normalized.loc[0, "description"] == normalized.loc[0, "clean_skill_evidence_text"]
