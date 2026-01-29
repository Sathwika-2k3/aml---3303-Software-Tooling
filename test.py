import pandas as pd


def load_csv(url: str) -> pd.DataFrame:
    """
    Load a CSV file from a URL into a Pandas DataFrame.

    """
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        raise ValueError(f"Failed to load CSV from {url}: {e}")


def calculate_average(df: pd.DataFrame, column: str) -> float:
    """
    Calculate the average of a numeric column in the DataFrame.

    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    return df[column].mean()


def calculate_max(df: pd.DataFrame, column: str) -> float:
    """
    Calculate the maximum value of a numeric column in the DataFrame.

    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    return df[column].max()


def filter_rows(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
    """
    Filter rows in the DataFrame where column equals the given value.

    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    return df[df[column] == value]

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    
    # Load data
    df = load_csv(url)
    
    # Compute statistics
    avg_sepal_length = calculate_average(df, "sepal_length")
    print("Average sepal length:", avg_sepal_length)
    
    max_petal_width = calculate_max(df, "petal_width")
    print("Max petal width:", max_petal_width)
    
    # Filter species
    setosa_rows = filter_rows(df, "species", "setosa")
    print(setosa_rows.head())
