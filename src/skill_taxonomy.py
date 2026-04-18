from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SkillDefinition:
    """Canonical enterprise skill definition used by the taxonomy."""

    skill_id: str
    canonical_name: str
    category: str
    description: str


@dataclass(frozen=True)
class SkillAliasDefinition:
    """Alias, tool, or related phrase that maps back to a canonical skill."""

    alias_id: str
    alias_text: str
    skill_id: str
    alias_type: str
    weight: float


@dataclass(frozen=True)
class SkillEdgeDefinition:
    """Lightweight graph edge used for small related-skill score propagation."""

    edge_id: str
    source_skill_id: str
    target_skill_id: str
    edge_type: str
    weight: float


@dataclass(frozen=True)
class SkillTaxonomy:
    """Container for the three enterprise taxonomy datasets."""

    skills: tuple[SkillDefinition, ...]
    aliases: tuple[SkillAliasDefinition, ...]
    edges: tuple[SkillEdgeDefinition, ...]

    @property
    def skill_by_id(self) -> dict[str, SkillDefinition]:
        return {skill.skill_id: skill for skill in self.skills}

    @property
    def skill_by_name(self) -> dict[str, SkillDefinition]:
        return {skill.canonical_name.lower(): skill for skill in self.skills}

    @property
    def aliases_by_skill_id(self) -> dict[str, tuple[SkillAliasDefinition, ...]]:
        grouped: dict[str, list[SkillAliasDefinition]] = {skill.skill_id: [] for skill in self.skills}
        for alias in self.aliases:
            grouped.setdefault(alias.skill_id, []).append(alias)
        return {skill_id: tuple(items) for skill_id, items in grouped.items()}

    @property
    def outgoing_edges(self) -> dict[str, tuple[SkillEdgeDefinition, ...]]:
        grouped: dict[str, list[SkillEdgeDefinition]] = {skill.skill_id: [] for skill in self.skills}
        for edge in self.edges:
            grouped.setdefault(edge.source_skill_id, []).append(edge)
        return {skill_id: tuple(items) for skill_id, items in grouped.items()}

    def skill_profiles(self) -> dict[str, str]:
        """Return the canonical text profiles used by TF-IDF scoring."""

        return {skill.canonical_name: skill.description for skill in self.skills}

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        """Expose the three taxonomy datasets as dataframes."""

        return {
            "skills": pd.DataFrame([skill.__dict__ for skill in self.skills]),
            "skill_aliases": pd.DataFrame([alias.__dict__ for alias in self.aliases]),
            "skill_edges": pd.DataFrame([edge.__dict__ for edge in self.edges]),
        }

    def export_csvs(self, output_dir: str | Path) -> dict[str, Path]:
        """Write the skills, aliases, and edges datasets to CSV files."""

        export_dir = Path(output_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        frames = self.to_dataframes()
        paths: dict[str, Path] = {}
        for name, frame in frames.items():
            output_path = export_dir / f"{name}.csv"
            frame.to_csv(output_path, index=False)
            paths[name] = output_path
        return paths


def build_default_skill_taxonomy() -> SkillTaxonomy:
    """Create the fixed enterprise taxonomy used by the MVP."""

    skills = (
        SkillDefinition(
            skill_id="SK001",
            canonical_name="Data Analysis",
            category="Analytics",
            description="Applies data analysis to explore datasets, identify patterns, evaluate trends, and produce actionable business insight.",
        ),
        SkillDefinition(
            skill_id="SK002",
            canonical_name="Data Governance",
            category="Governance",
            description="Applies data governance by improving data quality, defining standards, enforcing controls, and supporting trusted reporting.",
        ),
        SkillDefinition(
            skill_id="SK003",
            canonical_name="ETL Design",
            category="Data Engineering",
            description="Designs ETL processes to extract, transform, validate, and load data across systems in a controlled and scalable way.",
        ),
        SkillDefinition(
            skill_id="SK004",
            canonical_name="Power BI",
            category="Analytics",
            description="Uses Power BI to build dashboards, model measures, create visuals, and communicate performance insights to stakeholders.",
        ),
        SkillDefinition(
            skill_id="SK005",
            canonical_name="Programme Delivery",
            category="Delivery",
            description="Supports programme delivery through planning, governance, milestone tracking, workstream coordination, and issue management.",
        ),
        SkillDefinition(
            skill_id="SK006",
            canonical_name="Python",
            category="Engineering",
            description="Uses Python for scripting, automation, analysis, and technical problem solving across data and reporting workflows.",
        ),
        SkillDefinition(
            skill_id="SK007",
            canonical_name="Regulatory Reporting",
            category="Compliance",
            description="Supports regulatory reporting by preparing submissions, validating data, reconciling outputs, and meeting reporting obligations.",
        ),
        SkillDefinition(
            skill_id="SK008",
            canonical_name="Risk Controls",
            category="Risk",
            description="Applies risk controls through control design, testing, assurance, mitigation, and control effectiveness monitoring.",
        ),
        SkillDefinition(
            skill_id="SK009",
            canonical_name="SQL",
            category="Data Engineering",
            description="Uses SQL to query databases, join tables, transform data, optimize retrieval, and support reporting or analysis.",
        ),
        SkillDefinition(
            skill_id="SK010",
            canonical_name="Stakeholder Management",
            category="Delivery",
            description="Manages stakeholders by gathering requirements, aligning expectations, communicating decisions, and coordinating cross-functional delivery.",
        ),
    )

    aliases = (
        SkillAliasDefinition("AL001", "data analytics", "SK001", "synonym", 0.95),
        SkillAliasDefinition("AL002", "insight generation", "SK001", "related_phrase", 0.75),
        SkillAliasDefinition("AL003", "trend analysis", "SK001", "related_phrase", 0.78),
        SkillAliasDefinition("AL004", "exploratory analysis", "SK001", "related_phrase", 0.82),
        SkillAliasDefinition("AL005", "interpreting datasets", "SK001", "related_phrase", 0.72),
        SkillAliasDefinition("AL006", "data quality", "SK002", "related_phrase", 0.92),
        SkillAliasDefinition("AL007", "data stewardship", "SK002", "synonym", 0.87),
        SkillAliasDefinition("AL008", "data standards", "SK002", "subconcept", 0.82),
        SkillAliasDefinition("AL009", "governance framework", "SK002", "related_phrase", 0.78),
        SkillAliasDefinition("AL010", "master data controls", "SK002", "related_phrase", 0.72),
        SkillAliasDefinition("AL011", "etl pipeline", "SK003", "exact", 0.96),
        SkillAliasDefinition("AL012", "extract transform load", "SK003", "exact", 0.96),
        SkillAliasDefinition("AL013", "data integration", "SK003", "synonym", 0.86),
        SkillAliasDefinition("AL014", "ingestion pipeline", "SK003", "related_phrase", 0.82),
        SkillAliasDefinition("AL015", "workflow orchestration", "SK003", "related_phrase", 0.74),
        SkillAliasDefinition("AL016", "powerbi", "SK004", "exact", 0.99),
        SkillAliasDefinition("AL017", "dax", "SK004", "tool", 0.92),
        SkillAliasDefinition("AL018", "power query", "SK004", "tool", 0.88),
        SkillAliasDefinition("AL019", "dashboard reporting", "SK004", "related_phrase", 0.82),
        SkillAliasDefinition("AL020", "visual analytics", "SK004", "related_phrase", 0.76),
        SkillAliasDefinition("AL021", "program delivery", "SK005", "exact", 0.99),
        SkillAliasDefinition("AL022", "delivery management", "SK005", "synonym", 0.87),
        SkillAliasDefinition("AL023", "milestone tracking", "SK005", "related_phrase", 0.84),
        SkillAliasDefinition("AL024", "workstream coordination", "SK005", "related_phrase", 0.82),
        SkillAliasDefinition("AL025", "roadmap delivery", "SK005", "related_phrase", 0.77),
        SkillAliasDefinition("AL026", "python scripting", "SK006", "exact", 0.96),
        SkillAliasDefinition("AL027", "automation scripts", "SK006", "related_phrase", 0.84),
        SkillAliasDefinition("AL028", "pandas", "SK006", "tool", 0.91),
        SkillAliasDefinition("AL029", "numpy", "SK006", "tool", 0.87),
        SkillAliasDefinition("AL030", "jupyter notebook", "SK006", "tool", 0.78),
        SkillAliasDefinition("AL031", "regulatory submissions", "SK007", "synonym", 0.91),
        SkillAliasDefinition("AL032", "regulatory returns", "SK007", "synonym", 0.87),
        SkillAliasDefinition("AL033", "prudential reporting", "SK007", "related_phrase", 0.83),
        SkillAliasDefinition("AL034", "compliance reporting", "SK007", "related_phrase", 0.82),
        SkillAliasDefinition("AL035", "statutory reporting", "SK007", "related_phrase", 0.76),
        SkillAliasDefinition("AL036", "controls testing", "SK008", "related_phrase", 0.87),
        SkillAliasDefinition("AL037", "control framework", "SK008", "subconcept", 0.86),
        SkillAliasDefinition("AL038", "control effectiveness", "SK008", "related_phrase", 0.83),
        SkillAliasDefinition("AL039", "risk mitigation", "SK008", "related_phrase", 0.78),
        SkillAliasDefinition("AL040", "assurance controls", "SK008", "related_phrase", 0.73),
        SkillAliasDefinition("AL041", "structured query language", "SK009", "exact", 0.99),
        SkillAliasDefinition("AL042", "sql queries", "SK009", "exact", 0.96),
        SkillAliasDefinition("AL043", "joins", "SK009", "subconcept", 0.77),
        SkillAliasDefinition("AL044", "stored procedures", "SK009", "subconcept", 0.77),
        SkillAliasDefinition("AL045", "relational database", "SK009", "related_phrase", 0.76),
        SkillAliasDefinition("AL046", "stakeholder engagement", "SK010", "synonym", 0.91),
        SkillAliasDefinition("AL047", "requirements gathering", "SK010", "related_phrase", 0.87),
        SkillAliasDefinition("AL048", "client communication", "SK010", "related_phrase", 0.82),
        SkillAliasDefinition("AL049", "senior stakeholders", "SK010", "related_phrase", 0.78),
        SkillAliasDefinition("AL050", "cross functional coordination", "SK010", "related_phrase", 0.77),
    )

    edges = (
        SkillEdgeDefinition("ED001", "SK009", "SK001", "used_with", 0.85),
        SkillEdgeDefinition("ED002", "SK009", "SK004", "used_with", 0.80),
        SkillEdgeDefinition("ED003", "SK009", "SK007", "used_with", 0.70),
        SkillEdgeDefinition("ED004", "SK006", "SK001", "used_with", 0.80),
        SkillEdgeDefinition("ED005", "SK006", "SK003", "used_with", 0.75),
        SkillEdgeDefinition("ED006", "SK003", "SK002", "related_to", 0.65),
        SkillEdgeDefinition("ED007", "SK003", "SK009", "used_with", 0.85),
        SkillEdgeDefinition("ED008", "SK002", "SK007", "related_to", 0.80),
        SkillEdgeDefinition("ED009", "SK002", "SK008", "related_to", 0.75),
        SkillEdgeDefinition("ED010", "SK005", "SK010", "related_to", 0.85),
        SkillEdgeDefinition("ED011", "SK005", "SK007", "related_to", 0.60),
        SkillEdgeDefinition("ED012", "SK007", "SK008", "related_to", 0.75),
        SkillEdgeDefinition("ED013", "SK001", "SK010", "related_to", 0.55),
        SkillEdgeDefinition("ED014", "SK004", "SK010", "related_to", 0.60),
        SkillEdgeDefinition("ED015", "SK008", "SK010", "related_to", 0.55),
    )

    return SkillTaxonomy(skills=skills, aliases=aliases, edges=edges)


def export_default_taxonomy_csvs(output_dir: str | Path) -> dict[str, Path]:
    """Convenience wrapper for exporting the default taxonomy."""

    taxonomy = build_default_skill_taxonomy()
    return taxonomy.export_csvs(output_dir)
