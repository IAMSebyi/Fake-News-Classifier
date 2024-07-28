import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data from the given filepath."""
    return pd.read_csv(filepath)


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame to CSV."""
    df.to_csv(filepath, index=False)
