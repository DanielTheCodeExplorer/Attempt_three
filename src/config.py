from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ScoreBand:
    """Represents a similarity threshold and its label."""

    label: str
    minimum_similarity: float


@dataclass(frozen=True)
class PipelineConfig:
    """Central configuration for the competency scoring MVP."""

    input_csv_path: Path = Path("data/outputs/master_dataset.csv")
    output_csv_path: Path = Path("data/outputs/competency_scores.csv")
    evaluation_skill_summary_csv_path: Path = Path("data/outputs/competency_score_skill_summary.csv")
    evaluation_band_summary_csv_path: Path = Path("data/outputs/competency_score_band_summary.csv")
    evaluation_employee_summary_csv_path: Path = Path("data/outputs/competency_score_employee_summary.csv")
    evaluation_report_path: Path = Path("Implementation/competency_score_evaluation.md")
    required_columns: tuple[str, ...] = ("talentlinkId", "description", "skills")
    input_column_aliases: dict[str, str] = field(
        default_factory=lambda: {
            "TalentLink ID": "talentlinkId",
            "Description": "description",
            "skills": "skills",
        }
    )
    tfidf_max_features: int | None = 5000
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    score_bands: tuple[ScoreBand, ...] = (
        ScoreBand(label="low", minimum_similarity=0.0),
        ScoreBand(label="medium", minimum_similarity=0.2),
        ScoreBand(label="high", minimum_similarity=0.45),
    )
    default_skill_reference_template: str = (
        "Evidence of {skill} includes practical delivery using {skill} to solve problems, "
        "support implementation, analyse information, and contribute to business outcomes."
    )
    skill_references: dict[str, str] = field(
        default_factory=lambda: {
            "python": "Uses Python for scripting, automation, backend development, and data analysis.",
            "sql": "Uses SQL to query databases, join tables, transform data, and support reporting or analysis.",
            "data analysis": "Applies data analysis to interpret datasets, identify patterns, and produce actionable insights.",
            "power bi": "Uses Power BI to build dashboards, visualise data, and communicate performance insights.",
            "risk controls": "Applies risk controls to assess issues, strengthen governance, and improve control effectiveness.",
            "stakeholder management": "Manages stakeholders by gathering requirements, aligning priorities, and communicating progress clearly.",
            "etl design": "Designs ETL processes to extract, transform, validate, and load data across systems.",
            "programme delivery": "Supports programme delivery through planning, coordination, tracking actions, and driving milestones.",
            "regulatory reporting": "Supports regulatory reporting by preparing accurate submissions, validating data, and meeting reporting requirements.",
        }
    )
    excerpt_window_sentences: int = 1


DEFAULT_CONFIG = PipelineConfig()
