"""
preprocessor.py — Full preprocessing pipeline for the Airbnb NYC 2019 dataset.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are not useful for modeling."""
    cols_to_drop = ["id", "name", "host_id", "host_name"]
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)
    logger.info(f"Dropped columns: {existing}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values:
    - reviews_per_month: fill with 0 (listing has no reviews)
    - last_review: convert to days_since_last_review (numerical feature)
    """
    if "reviews_per_month" in df.columns:
        df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
        logger.info("Filled missing reviews_per_month with 0.")

    if "last_review" in df.columns:
        df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
        reference_date = df["last_review"].max()
        df["days_since_last_review"] = (reference_date - df["last_review"]).dt.days
        max_days = df["days_since_last_review"].max()
        df["days_since_last_review"] = df["days_since_last_review"].fillna(max_days)
        df = df.drop(columns=["last_review"])
        logger.info("Converted last_review → days_since_last_review.")

    return df


def remove_price_outliers(df: pd.DataFrame, min_price: float = 10, max_price: float = 1000) -> pd.DataFrame:
    """Remove listings with extreme price values."""
    before = len(df)
    df = df[(df["price"] >= min_price) & (df["price"] <= max_price)]
    logger.info(f"Removed {before - len(df)} price outlier rows. Kept price range: ${min_price}–${max_price}.")
    return df


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features. drop_first=True avoids multicollinearity."""
    cat_cols = ["neighbourhood_group", "neighbourhood", "room_type"]
    existing = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing, drop_first=True)
    logger.info(f"One-hot encoded: {existing}")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline.

    Steps:
        1. Drop unnecessary columns
        2. Handle missing values
        3. Remove price outliers
        4. Encode categorical variables

    Args:
        df: Raw DataFrame.

    Returns:
        pd.DataFrame: Cleaned and encoded DataFrame ready for modeling.
    """
    logger.info("Starting preprocessing pipeline...")
    df = drop_unnecessary_columns(df)
    df = handle_missing_values(df)
    df = remove_price_outliers(df)
    df = encode_categorical_columns(df)
    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    return df


def split_features_target(df: pd.DataFrame, target_col: str = "price"):
    """
    Split the DataFrame into features (X) and target (y).

    Args:
        df: Preprocessed DataFrame.
        target_col: Name of the target column.

    Returns:
        X (pd.DataFrame), y (pd.Series)
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    logger.info(f"Features: {X.shape}, Target: {y.shape}")
    return X, y
