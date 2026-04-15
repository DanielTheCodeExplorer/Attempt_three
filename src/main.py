from src.competency_scoring import CompetencyScorer
from src.config import DEFAULT_CONFIG
from src.data_loader import load_and_prepare_dataset
from src.logging_utils import configure_logging, get_logger


LOGGER = get_logger(__name__)


def main() -> None:
    """Run the MVP competency scoring pipeline end to end."""

    configure_logging()
    dataset = load_and_prepare_dataset(DEFAULT_CONFIG.input_csv_path, DEFAULT_CONFIG)
    scorer = CompetencyScorer(DEFAULT_CONFIG)
    results = scorer.score_dataframe(dataset)
    DEFAULT_CONFIG.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(DEFAULT_CONFIG.output_csv_path, index=False)
    LOGGER.info("pipeline_complete output=%s rows=%s", DEFAULT_CONFIG.output_csv_path, len(results))


if __name__ == "__main__":
    main()
