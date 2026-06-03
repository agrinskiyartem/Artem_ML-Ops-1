import logging
from pathlib import Path

import pandas as pd

from src.config import RAW_FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def validate_input_columns(df: pd.DataFrame) -> None:
    """Validate that all columns required for inference are present."""
    missing_columns = [col for col in RAW_FEATURE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Input file is missing required columns: "
            f"{missing_columns}. Expected columns: {RAW_FEATURE_COLUMNS}"
        )

    extra_columns = [col for col in df.columns if col not in RAW_FEATURE_COLUMNS and col != "target"]
    if extra_columns:
        logger.warning("Input file contains extra columns that will be ignored: %s", extra_columns)


def load_input_data(path: Path) -> pd.DataFrame:
    """Load the mounted test.csv file for batch inference."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Input file was not found: {path}. "
            "Mount a directory with test.csv to /app/input or place test.csv in input/."
        )

    logger.info("Loading input data from %s", path)
    df = pd.read_csv(path)
    validate_input_columns(df)
    logger.info("Loaded input data with shape %s", df.shape)
    return df
