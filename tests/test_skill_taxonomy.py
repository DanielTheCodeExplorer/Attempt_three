from pathlib import Path
import json

import pandas as pd

from src.competency_scoring import CompetencyScorer
from src.config import PipelineConfig
from src.skill_extraction import extract_skills
from src.skill_taxonomy import build_default_skill_taxonomy, export_default_taxonomy_csvs


def test_default_skill_taxonomy_has_expected_sizes():
    taxonomy = build_default_skill_taxonomy()

    assert len(taxonomy.skills) == 10
    assert 30 <= len(taxonomy.aliases) <= 60
    assert 10 <= len(taxonomy.edges) <= 20


def test_extract_skills_returns_alias_hits_and_evidence():
    taxonomy = build_default_skill_taxonomy()

    payload = extract_skills(
        "Improved data quality controls and built dashboards in Power BI for senior stakeholders.",
        taxonomy,
        config=PipelineConfig(),
    )

    skill_names = {item["canonical_name"] for item in payload["matched_skills"]}
    assert "Data Governance" in skill_names
    assert "Power BI" in skill_names
    assert payload["matched_aliases"]
    assert payload["evidence_snippets"]


def test_export_default_taxonomy_csvs_writes_three_csvs(tmp_path: Path):
    exported = export_default_taxonomy_csvs(tmp_path)

    assert set(exported) == {"skills", "skill_aliases", "skill_edges"}
    assert all(path.exists() for path in exported.values())

    skills_df = pd.read_csv(exported["skills"])
    aliases_df = pd.read_csv(exported["skill_aliases"])
    edges_df = pd.read_csv(exported["skill_edges"])

    assert len(skills_df) == 10
    assert 30 <= len(aliases_df) <= 60
    assert 10 <= len(edges_df) <= 20


def test_competency_scorer_exports_taxonomy_csvs(tmp_path: Path):
    scorer = CompetencyScorer(PipelineConfig())

    exported = scorer.export_taxonomy_csvs(str(tmp_path))

    assert set(exported) == {"skills", "skill_aliases", "skill_edges"}
    assert all(Path(path).exists() for path in exported.values())
