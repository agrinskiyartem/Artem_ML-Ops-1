from pathlib import Path

APP_DIR = Path("/app")
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Runtime paths inside Docker. They are also usable locally when the project is
# copied to /app; local scripts can override them through function arguments.
INPUT_PATH = APP_DIR / "input" / "test.csv"
OUTPUT_PATH = APP_DIR / "output" / "sample_submission.csv"
MODEL_PATH = APP_DIR / "models" / "model.joblib"

LOCAL_MODEL_PATH = PROJECT_DIR / "models" / "model.joblib"
TRAIN_PATH = PROJECT_DIR / "train.csv"

SAMPLE_SUBMISSION_CANDIDATES = (
    PROJECT_DIR / "sample_submission.csv",
    PROJECT_DIR / "sample_submition.csv",  # common typo in the competition files
    APP_DIR / "sample_submission.csv",
    APP_DIR / "sample_submition.csv",
)

RAW_FEATURE_COLUMNS = [
    "transaction_time",
    "merch",
    "cat_id",
    "amount",
    "name_1",
    "name_2",
    "gender",
    "street",
    "one_city",
    "us_state",
    "post_code",
    "lat",
    "lon",
    "population_city",
    "jobs",
    "merchant_lat",
    "merchant_lon",
]

TARGET_COLUMN = "target"

NUMERIC_FEATURES = [
    "amount",
    "lat",
    "lon",
    "population_city",
    "merchant_lat",
    "merchant_lon",
    "transaction_hour",
    "transaction_dayofweek",
    "transaction_month",
    "transaction_day",
    "is_weekend",
    "distance_km",
]

CATEGORICAL_FEATURES = [
    "merch",
    "cat_id",
    "gender",
    "one_city",
    "us_state",
    "post_code",
    "jobs",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
