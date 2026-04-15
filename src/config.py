from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ScoreBand:
    """Represents a similarity threshold and its label."""

    label: str
    minimum_score: float


@dataclass(frozen=True)
class PipelineConfig:
    """Central configuration for the competency scoring MVP."""

    input_csv_path: Path = Path("data/outputs/master_dataset.csv")
    output_csv_path: Path = Path("data/outputs/competency_scores.csv")
    biography_input_csv_path: Path = Path("data/outputs/master_dataset.csv")
    biography_dataset_csv_path: Path = Path("data/outputs/biography_dataset.csv")
    biography_score_output_csv_path: Path = Path("data/outputs/biography_competency_scores.csv")
    biography_evaluation_csv_path: Path = Path("data/outputs/biography_competency_evaluation.csv")
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
            "Biography": "description",
            "biography": "description",
        }
    )
    tfidf_max_features: int | None = 5000
    tfidf_ngram_range: tuple[int, int] = (1, 3)
    tfidf_stop_words: str | None = "english"
    tfidf_sublinear_tf: bool = True
    exact_match_weight: float = 0.6
    semantic_similarity_weight: float = 0.4
    semantic_similarity_strong_threshold: float = 0.12
    score_bands: tuple[ScoreBand, ...] = (
        ScoreBand(label="low", minimum_score=0.0),
        ScoreBand(label="medium", minimum_score=25.0),
        ScoreBand(label="high", minimum_score=60.0),
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
    skill_aliases: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "python": ("pandas", "pyspark", "jupyter", "python scripting", "python automation"),
            "sql": ("structured query language", "relational databases", "database querying", "sql queries", "joins"),
            "data analysis": ("data analytics", "analyse data", "analyze data", "data insights", "pattern analysis"),
            "power bi": ("powerbi", "dashboard reporting", "dashboards", "business intelligence reporting"),
            "risk controls": ("control effectiveness", "controls testing", "control framework", "risk governance"),
            "stakeholder management": ("stakeholder engagement", "stakeholder communication", "requirements gathering", "cross functional collaboration"),
            "etl design": ("etl pipeline", "data pipelines", "extract transform load", "data integration"),
            "programme delivery": ("program delivery", "project delivery", "workstream delivery", "milestone tracking", "delivery coordination"),
            "regulatory reporting": ("regulatory submissions", "regulatory returns", "compliance reporting", "reporting requirements"),
            "data governance": ("data quality", "data stewardship", "data standards", "governance framework"),
        }
    )
    excerpt_window_sentences: int = 1


DEFAULT_CONFIG = PipelineConfig()
