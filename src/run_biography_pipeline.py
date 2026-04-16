from src.biography_dataset import create_biography_dataset
from src.competency_scoring import CompetencyScorer
from src.config import DEFAULT_CONFIG
from src.data_loader import load_and_prepare_dataset
from src.evaluation import summarise_scores
from src.llm_normalizer import normalize_biography_dataframe
from src.logging_utils import configure_logging, get_logger
import pandas as pd


LOGGER = get_logger(__name__)


def main() -> None:
    """Create a biography dataset, score it, and write a biography evaluation CSV."""

    configure_logging()
    create_biography_dataset(
        input_csv_path=DEFAULT_CONFIG.biography_input_csv_path,
        output_csv_path=DEFAULT_CONFIG.biography_dataset_csv_path,
    )
    biography_df = pd.read_csv(DEFAULT_CONFIG.biography_dataset_csv_path)
    normalized_biography_df = normalize_biography_dataframe(biography_df, config=DEFAULT_CONFIG)
    normalized_biography_df.to_csv(DEFAULT_CONFIG.biography_normalized_dataset_csv_path, index=False)

    dataset = load_and_prepare_dataset(DEFAULT_CONFIG.biography_normalized_dataset_csv_path, DEFAULT_CONFIG)
    scorer = CompetencyScorer(DEFAULT_CONFIG)
    results = scorer.score_dataframe(dataset)

    DEFAULT_CONFIG.biography_score_output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(DEFAULT_CONFIG.biography_score_output_csv_path, index=False)

    evaluation_df = summarise_scores(results)
    evaluation_df.to_csv(DEFAULT_CONFIG.biography_evaluation_csv_path, index=False)

    LOGGER.info(
        "biography_pipeline_complete dataset=%s scores=%s evaluation=%s",
        DEFAULT_CONFIG.biography_normalized_dataset_csv_path,
        DEFAULT_CONFIG.biography_score_output_csv_path,
        DEFAULT_CONFIG.biography_evaluation_csv_path,
    )


if __name__ == "__main__":
    main()
