from src.competency_scoring import CompetencyScorer
from src.config import DEFAULT_CONFIG
from src.data_loader import load_and_prepare_dataset
from src.logging_utils import configure_logging, get_logger


LOGGER = get_logger(__name__)


def main() -> None:
    """Score the job-history route and write the main score output."""

    configure_logging()
    dataset = load_and_prepare_dataset(DEFAULT_CONFIG.input_csv_path, DEFAULT_CONFIG)
    scorer = CompetencyScorer(DEFAULT_CONFIG)
    exported = scorer.export_taxonomy_csvs()
    results = scorer.score_dataframe(dataset)

    DEFAULT_CONFIG.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(DEFAULT_CONFIG.output_csv_path, index=False)

    LOGGER.info(
        "job_history_pipeline_complete scores=%s rows=%s taxonomy=%s",
        DEFAULT_CONFIG.output_csv_path,
        len(results),
        exported,
    )


if __name__ == "__main__":
    main()
