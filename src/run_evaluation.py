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
    """Run evaluation over the competency scoring output and write a markdown report."""

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

    Path(DEFAULT_CONFIG.evaluation_report_path).parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_CONFIG.evaluation_report_path.write_text(report)

    LOGGER.info(
        "evaluation_complete report=%s",
        DEFAULT_CONFIG.evaluation_report_path,
    )


if __name__ == "__main__":
    main()
