from src.config import DEFAULT_CONFIG
from src.final_competency_dataset import build_final_competency_dataset
from src.logging_utils import configure_logging


def main() -> None:
    """Build the final consolidated competency dataset."""

    configure_logging()
    build_final_competency_dataset(DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
