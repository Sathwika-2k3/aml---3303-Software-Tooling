import pandas as pd

def preprocess_data(df):
    """Preprocess the Airbnb data."""
    # Drop unnecessary columns
    df = df.drop(['id', 'name', 'host_id', 'host_name'], axis=1)

    # Handle missing values
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    df['days_since_last_review'] = (pd.Timestamp.now() - df['last_review']).dt.days
    df['days_since_last_review'] = df['days_since_last_review'].fillna(df['days_since_last_review'].max())
    df = df.drop('last_review', axis=1)

    # Remove outliers
    df = df[(df['price'] > 10) & (df['price'] < 1000)]

    # Encode categoricals
    df = pd.get_dummies(df, columns=['neighbourhood_group', 'neighbourhood', 'room_type'], drop_first=True)

    return df
