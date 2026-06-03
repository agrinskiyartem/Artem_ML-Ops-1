import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_model(model_path: Path):
    """Load the pre-trained model artifact used only for inference."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact was not found: {model_path}. "
            "Run `python train_model.py` before building the Docker image so models/model.joblib exists."
        )

    logger.info("Loading model artifact from %s", model_path)
    return joblib.load(model_path)


def score_data(model, features: pd.DataFrame) -> np.ndarray:
    """Return fraud scores. Prefer probability of class 1 when available."""
    logger.info("Scoring %s rows", len(features))
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities.ravel()

    predictions = model.predict(features)
    return np.asarray(predictions).ravel()
