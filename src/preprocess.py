import logging

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, RAW_FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


def _haversine_distance_km(lat1, lon1, lat2, lon2):
    """Calculate vectorized haversine distance in kilometers."""
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic inference features without fitting anything."""
    logger.info("Starting preprocessing for shape %s", df.shape)
    data = df.copy()

    if TARGET_COLUMN in data.columns:
        data = data.drop(columns=[TARGET_COLUMN])

    # Keep only known raw columns. Extra input columns are intentionally ignored.
    data = data[RAW_FEATURE_COLUMNS]

    transaction_time = pd.to_datetime(data["transaction_time"], errors="coerce")
    data["transaction_hour"] = transaction_time.dt.hour
    data["transaction_dayofweek"] = transaction_time.dt.dayofweek
    data["transaction_month"] = transaction_time.dt.month
    data["transaction_day"] = transaction_time.dt.day
    data["is_weekend"] = data["transaction_dayofweek"].isin([5, 6]).astype(int)

    coordinate_columns = ["lat", "lon", "merchant_lat", "merchant_lon"]
    for column in coordinate_columns + ["amount", "population_city"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["distance_km"] = _haversine_distance_km(
        data["lat"], data["lon"], data["merchant_lat"], data["merchant_lon"]
    )

    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].astype("string").fillna("unknown")

    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    processed = data[MODEL_FEATURES]
    logger.info("Preprocessing completed. Output shape: %s", processed.shape)
    return processed
