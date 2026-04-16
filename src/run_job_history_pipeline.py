from src.competency_scoring import CompetencyScorer
from src.config import DEFAULT_CONFIG
from src.data_loader import load_and_prepare_dataset
from src.evaluation import build_pipeline_evaluation_frame
from src.logging_utils import configure_logging, get_logger


LOGGER = get_logger(__name__)


def main() -> None:
    """Score the job-history route and write job-history-only outputs."""

    configure_logging()
    dataset = load_and_prepare_dataset(DEFAULT_CONFIG.input_csv_path, DEFAULT_CONFIG)
    scorer = CompetencyScorer(DEFAULT_CONFIG)
    results = scorer.score_dataframe(dataset)

    DEFAULT_CONFIG.job_history_score_output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(DEFAULT_CONFIG.job_history_score_output_csv_path, index=False)

    evaluation_df = build_pipeline_evaluation_frame(results, data_source="job_history")
    evaluation_df.to_csv(DEFAULT_CONFIG.job_history_evaluation_csv_path, index=False)

    LOGGER.info(
        "job_history_pipeline_complete scores=%s evaluation=%s rows=%s",
        DEFAULT_CONFIG.job_history_score_output_csv_path,
        DEFAULT_CONFIG.job_history_evaluation_csv_path,
        len(results),
    )


if __name__ == "__main__":
    main()
