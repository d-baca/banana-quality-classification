import pandas as pd
import numpy as np
from src.exploratory_data_analysis.load_csv_data import load_csv_data


def describe_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zwraca opis statystyczny danych w DataFrame.
    """
    if df.empty:
        print("DataFrame is empty. No description available.")
        return pd.DataFrame()

    nums = df.select_dtypes(include=[np.number])
    description = nums.describe().T  # Transpose dla lepszej czytelności
    description["skew"] = nums.skew()
    description["kurtosis"] = nums.kurtosis()
    return description


def save_description(description: pd.DataFrame, out_path: str = "reports/figures/descriptive_stats.csv"):
    """
    Zapisuje opis statystyczny do pliku CSV.
    """
    try:
        description.to_csv(out_path, index=True)
        print(f"Description saved successfully to {out_path}")
    except Exception as e:
        print(f"An error occurred while saving the description: {e}")


if __name__ == "__main__":
    df = load_csv_data()
    if not df.empty:
        description = describe_data(df.select_dtypes(include='number'))
        save_description(description)
        print(description)
    else:
        print("No data to describe.")
