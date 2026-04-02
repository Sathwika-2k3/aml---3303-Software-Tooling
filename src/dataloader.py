import pandas as pd

def load_data(file_path):
    """Load Airbnb data from S3."""
    return pd.read_csv(file_path)
