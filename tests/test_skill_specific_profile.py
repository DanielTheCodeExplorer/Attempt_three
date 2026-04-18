import json

from src.config import PipelineConfig
from src.skill_specific_profile import (
    SKILL_SPECIFIC_PROFILE_PROMPT,
    _build_skill_specific_messages,
    generate_skill_specific_profile,
)


class FakeHallucinationClient:
    class chat:
        class completions:
            @staticmethod
            def create(*args, **kwargs):
                payload = {
                    "skill_specific_profile": (
                        "python architect with aws certification and led a fraud platform migration"
                    )
                }
                return type(
                    "Response",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {"message": type("Message", (), {"content": json.dumps(payload)})()},
                            )()
                        ]
                    },
                )()


def test_skill_specific_profile_prompt_requires_reference_structure_alignment():
    assert "mirror the structure and style of the skill reference sentence" in SKILL_SPECIFIC_PROFILE_PROMPT
    assert "structural template" in SKILL_SPECIFIC_PROFILE_PROMPT
    assert "omit it rather than inventing" in SKILL_SPECIFIC_PROFILE_PROMPT


def test_build_skill_specific_messages_instructs_reference_shaped_output():
    messages = _build_skill_specific_messages(
        evidence_spans=["Built Python automation for reporting."],
        target_skill="Python",
        config=PipelineConfig(
            skill_references={"python": "Uses Python for scripting automation and data analysis."}
        ),
    )

    assert len(messages) == 2
    assert "Skill reference:\nUses Python for scripting automation and data analysis." in messages[1]["content"]
    assert "follows the reference sentence structure" in messages[1]["content"]


def test_generate_skill_specific_profile_keeps_strong_evidence_focused():
    config = PipelineConfig(
        skill_focused_rewrite_enabled=False,
        skill_references={
            "python": "Uses Python for scripting automation and data analysis.",
        },
    )

    payload = generate_skill_specific_profile(
        talentlink_id="1001",
        description=(
            "Built Python automation for regulatory reporting. "
            "Developed Python data analysis workflows for recurring audit checks. "
            "Planned office events and catering budgets."
        ),
        skills=["Python", "SQL"],
        target_skill="Python",
        config=config,
    )

    assert payload["talentlinkId"] == "1001"
    assert payload["target_skill"] == "Python"
    assert len(payload["evidence_spans"]) == 2
    assert "planned office events" not in payload["skill_specific_profile"]
    assert "python automation" in payload["skill_specific_profile"]
    assert payload["confidence_score"] >= 0.75
    assert "evidence related to python" in payload["scoring_text_used"]
    assert "uses python for scripting automation" in payload["scoring_text_used"]


def test_generate_skill_specific_profile_stays_cautious_with_weak_evidence():
    config = PipelineConfig(
        skill_focused_rewrite_enabled=False,
        skill_references={
            "sql": "Uses SQL to query tables, join data, and support reporting.",
        },
        skill_aliases={
            "sql": ("joins",),
        },
    )

    payload = generate_skill_specific_profile(
        talentlink_id="1002",
        description="Supported reporting tasks with joins in monthly reconciliations.",
        skills=["SQL"],
        target_skill="SQL",
        config=config,
    )

    assert payload["evidence_spans"] == [
        "Supported reporting tasks with joins in monthly reconciliations."
    ]
    assert payload["confidence_score"] < 0.75
    assert "joins in monthly reconciliations" in payload["skill_specific_profile"]
    assert "evidence related to sql" in payload["scoring_text_used"]
    assert "support reporting" in payload["scoring_text_used"]


def test_generate_skill_specific_profile_returns_empty_for_unsupported_skill():
    config = PipelineConfig(
        skill_focused_rewrite_enabled=False,
        skill_references={
            "python": "Uses Python for scripting automation and data analysis.",
        },
    )

    payload = generate_skill_specific_profile(
        talentlink_id="1003",
        description="Managed office moves and coordinated vendor invoices.",
        skills=["Python"],
        target_skill="Python",
        config=config,
    )

    assert payload["skill_specific_profile"] == ""
    assert payload["scoring_text_used"] == ""
    assert payload["evidence_spans"] == []
    assert payload["confidence_score"] == 0.0


def test_generate_skill_specific_profile_rejects_hallucinated_rewrite():
    config = PipelineConfig(
        skill_focused_rewrite_enabled=True,
        skill_references={
            "python": "Uses Python for scripting automation and data analysis.",
        },
    )

    payload = generate_skill_specific_profile(
        talentlink_id="1004",
        description="Built Python automation for reporting and reconciliations.",
        skills=["Python"],
        target_skill="Python",
        config=config,
        client=FakeHallucinationClient(),
    )

    assert payload["skill_specific_profile"] == "built python automation for reporting and reconciliations"
    assert "evidence related to python" in payload["scoring_text_used"]
    assert "aws certification" not in payload["skill_specific_profile"]
    assert "architect" not in payload["skill_specific_profile"]
    assert "fraud platform" not in payload["skill_specific_profile"]
