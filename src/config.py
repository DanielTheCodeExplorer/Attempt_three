import os
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
    job_history_score_output_csv_path: Path = Path("data/outputs/job_history_competency_scores.csv")
    job_history_evaluation_csv_path: Path = Path("data/outputs/job_history_competency_evaluation.csv")
    biography_input_csv_path: Path = Path("data/outputs/master_dataset.csv")
    biography_dataset_csv_path: Path = Path("data/outputs/biography_dataset.csv")
    biography_normalized_dataset_csv_path: Path = Path("data/outputs/biography_normalized_dataset.csv")
    biography_score_output_csv_path: Path = Path("data/outputs/biography_competency_scores.csv")
    final_competency_dataset_csv_path: Path = Path("data/outputs/final_competency_dataset.csv")
    biography_evaluation_csv_path: Path = Path("data/outputs/biography_competency_evaluation.csv")
    skill_specific_profile_output_csv_path: Path = Path("data/outputs/skill_specific_profiles.csv")
    taxonomy_output_dir: Path = Path("data/outputs/taxonomy")
    evaluation_skill_summary_csv_path: Path = Path("data/outputs/competency_score_skill_summary.csv")
    evaluation_band_summary_csv_path: Path = Path("data/outputs/competency_score_band_summary.csv")
    evaluation_employee_summary_csv_path: Path = Path("data/outputs/competency_score_employee_summary.csv")
    evaluation_report_path: Path = Path("Implementation/competency_score_evaluation.md")
    required_columns: tuple[str, ...] = ("talentlinkId", "description", "skills")
    input_column_aliases: dict[str, str] = field(
        default_factory=lambda: {
            "TalentLink ID": "talentlinkId",
            "Description": "description",
            "clean_skill_evidence_text": "description",
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
    sentence_transformer_model_name: str = field(
        default_factory=lambda: os.getenv("SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embedding_similarity_threshold: float = 0.50
    tfidf_weight_with_alias: float = 0.70
    alias_weight_in_final: float = 0.30
    tfidf_weight_with_embedding: float = 0.60
    embedding_weight_in_final: float = 0.40
    graph_boost_factor: float = 0.20
    minimum_output_skill_score: float = 0.12
    llm_model_name: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"))
    llm_base_url: str | None = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))
    skill_focused_rewrite_enabled: bool = field(
        default_factory=lambda: os.getenv("OPENAI_SKILL_FOCUSED_REWRITE", "1").strip().lower()
        not in {"0", "false", "no"}
    )
    skill_profile_top_k_chunks: int = 3
    skill_profile_min_chunk_score: float = 0.03
    skill_profile_min_llm_confidence: float = 0.35
    skill_profile_reference_alignment_min_confidence: float = 0.4
    skill_profile_reference_skill_prefix_min_confidence: float = 0.55
    demonstrated_skill_top_k: int = 3
    demonstrated_skill_min_score: float = 0.08
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
