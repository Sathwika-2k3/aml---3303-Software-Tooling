"""
dataloader.py — Loads the Airbnb NYC 2019 dataset from a public URL.
"""

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#AIRBNB_DATA_URL = "https://raw.githubusercontent.com/rashida048/Datasets/master/AB_NYC_2019.csv"
AIRBNB_DATA_URL = "data/AB_NYC_2019.csv"

def load_data(url: str = AIRBNB_DATA_URL) -> pd.DataFrame:
    """
    Load the Airbnb NYC 2019 dataset from a public URL.

    Args:
        url: URL pointing to the CSV file.

    Returns:
        pd.DataFrame: Raw loaded dataset.
    """
    try:
        logger.info(f"Loading data from: {url}")
        df = pd.read_csv(url)
        logger.info(f"Data loaded successfully — {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    except Exception as e:
        raise ValueError(f"Could not load data from {url}: {e}")


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Check that all required columns are present in the DataFrame.

    Args:
        df: DataFrame to check.
        required_columns: List of expected column names.

    Returns:
        True if all columns present, raises ValueError otherwise.
    """
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("DataFrame validation passed.")
    return True
