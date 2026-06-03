import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, LOCAL_MODEL_PATH, NUMERIC_FEATURES, TARGET_COLUMN, TRAIN_PATH
from src.load_data import validate_input_columns
from src.preprocess import preprocess_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def _check_lfs_pointer(train_path: Path) -> None:
    """Fail with a clear message when train.csv is only a Git LFS pointer."""
    with train_path.open("r", encoding="utf-8", errors="ignore") as file:
        first_line = file.readline().strip()
    if first_line == "version https://git-lfs.github.com/spec/v1":
        raise ValueError(
            "train.csv is a Git LFS pointer, not the real training dataset. "
            "Download or pull the actual train.csv before running training."
        )


def build_estimator(y: pd.Series) -> Pipeline:
    """Build a simple CPU-only sklearn estimator."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10, max_categories=100),
            ),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    if y.nunique(dropna=True) < 2:
        classifier = DummyClassifier(strategy="prior")
    else:
        classifier = LogisticRegression(max_iter=300, class_weight="balanced", n_jobs=1)

    return Pipeline(steps=[("features", transformer), ("classifier", classifier)])


def train_model(train_path: Path = TRAIN_PATH, model_path: Path = LOCAL_MODEL_PATH) -> None:
    """Train and save the model artifact used by the Docker inference service."""
    train_path = Path(train_path)
    model_path = Path(model_path)

    if not train_path.exists():
        raise FileNotFoundError(f"Training file was not found: {train_path}")

    _check_lfs_pointer(train_path)
    logger.info("Loading training data from %s", train_path)
    train_df = pd.read_csv(train_path)

    validate_input_columns(train_df)
    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Training data must contain target column: {TARGET_COLUMN}")

    y = train_df[TARGET_COLUMN].astype(int)
    features = preprocess_data(train_df.drop(columns=[TARGET_COLUMN]))

    estimator = build_estimator(y)
    logger.info("Fitting model on %s rows and %s features", features.shape[0], features.shape[1])
    estimator.fit(features, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, model_path)
    logger.info("Saved model artifact to %s", model_path)


if __name__ == "__main__":
    train_model()
