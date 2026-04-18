from src.biography_dataset import create_biography_dataset
from src.competency_scoring import CompetencyScorer
from src.config import DEFAULT_CONFIG
from src.llm_normalizer import normalize_biography_dataframe
from src.logging_utils import configure_logging, get_logger


LOGGER = get_logger(__name__)


def main() -> None:
    """Create a biography dataset, score it, and write biography-only outputs."""

    configure_logging()
    biography_df = create_biography_dataset(
        input_csv_path=DEFAULT_CONFIG.biography_input_csv_path,
        output_csv_path=None,
    )
    normalized_biography_df = normalize_biography_dataframe(biography_df, config=DEFAULT_CONFIG)
    normalized_biography_df.to_csv(DEFAULT_CONFIG.biography_normalized_dataset_csv_path, index=False)

    scorer = CompetencyScorer(DEFAULT_CONFIG)
    exported = scorer.export_taxonomy_csvs()
    results = scorer.score_dataframe(normalized_biography_df[["talentlinkId", "description", "skills"]])

    DEFAULT_CONFIG.biography_score_output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(DEFAULT_CONFIG.biography_score_output_csv_path, index=False)

    LOGGER.info(
        "biography_pipeline_complete normalized_dataset=%s scores=%s",
        DEFAULT_CONFIG.biography_normalized_dataset_csv_path,
        DEFAULT_CONFIG.biography_score_output_csv_path,
    )
    LOGGER.info("taxonomy_export_complete files=%s", exported)


if __name__ == "__main__":
    main()
