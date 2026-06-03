import logging
from pathlib import Path

import pandas as pd

from src.config import SAMPLE_SUBMISSION_CANDIDATES

logger = logging.getLogger(__name__)


def find_sample_submission() -> Path:
    """Find the competition submission template, including the typo variant."""
    for candidate in SAMPLE_SUBMISSION_CANDIDATES:
        if candidate.exists():
            return candidate
    candidates = ", ".join(str(path) for path in SAMPLE_SUBMISSION_CANDIDATES)
    raise FileNotFoundError(
        "Sample submission template was not found. Expected one of: " f"{candidates}"
    )


def save_submission(predictions, output_path: Path, template_path: Path | None = None) -> pd.DataFrame:
    """Write predictions to the target column while preserving template columns."""
    template_path = Path(template_path) if template_path else find_sample_submission()
    logger.info("Loading submission template from %s", template_path)
    submission = pd.read_csv(template_path)

    if len(submission) != len(predictions):
        raise ValueError(
            "Prediction length does not match sample submission length: "
            f"{len(predictions)} predictions vs {len(submission)} template rows."
        )

    if len(submission.columns) < 1:
        raise ValueError("Sample submission template must contain at least one column.")

    target_column = submission.columns[-1]
    submission[target_column] = predictions

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    logger.info("Saved submission with shape %s to %s", submission.shape, output_path)
    return submission
