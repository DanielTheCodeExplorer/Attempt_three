from pathlib import Path

import pandas as pd

from src.config import DEFAULT_CONFIG
from src.evaluation import (
    build_overall_metrics,
    render_evaluation_report,
    summarise_employee_scores,
    summarise_score_bands,
    summarise_scores,
)
from src.logging_utils import configure_logging, get_logger


LOGGER = get_logger(__name__)


def main() -> None:
    """Run evaluation over the competency scoring output and write summary artifacts."""

    configure_logging()
    scores_path = DEFAULT_CONFIG.output_csv_path
    if not scores_path.exists():
        raise FileNotFoundError(f"Score file not found: {scores_path}")

    results_df = pd.read_csv(scores_path)
    metrics = build_overall_metrics(results_df)
    skill_summary = summarise_scores(results_df)
    band_summary = summarise_score_bands(results_df)
    employee_summary = summarise_employee_scores(results_df)
    report = render_evaluation_report(metrics, skill_summary, band_summary, employee_summary)

    output_paths = [
        DEFAULT_CONFIG.evaluation_skill_summary_csv_path,
        DEFAULT_CONFIG.evaluation_band_summary_csv_path,
        DEFAULT_CONFIG.evaluation_employee_summary_csv_path,
        DEFAULT_CONFIG.evaluation_report_path,
    ]
    for path in output_paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    skill_summary.to_csv(DEFAULT_CONFIG.evaluation_skill_summary_csv_path, index=False)
    band_summary.to_csv(DEFAULT_CONFIG.evaluation_band_summary_csv_path, index=False)
    employee_summary.to_csv(DEFAULT_CONFIG.evaluation_employee_summary_csv_path, index=False)
    DEFAULT_CONFIG.evaluation_report_path.write_text(report)

    LOGGER.info(
        "evaluation_complete skill_summary=%s band_summary=%s employee_summary=%s report=%s",
        DEFAULT_CONFIG.evaluation_skill_summary_csv_path,
        DEFAULT_CONFIG.evaluation_band_summary_csv_path,
        DEFAULT_CONFIG.evaluation_employee_summary_csv_path,
        DEFAULT_CONFIG.evaluation_report_path,
    )


if __name__ == "__main__":
    main()
