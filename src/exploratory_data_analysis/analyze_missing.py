import pandas as pd
import os
from src.exploratory_data_analysis.load_csv_data import load_csv_data


def count_missing(df: pd.DataFrame) -> pd.Series:
    """
    Zlicza brakujące wartości w każdej kolumnie DataFrame.
    """
    return df.isnull().sum()


def report_missing(series: pd.Series, out_path="reports/figures/missing_report.txt"):
    """
    Tworzy raport z brakujących wartości i zapisuje go do pliku tekstowego.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for col, count in series.items():
            f.write(f"{col}: {count} missing values\n")
    print(f"Raport z brakujących wartości zapisany do {out_path}")


if __name__ == "__main__":
    df = load_csv_data()
    if not df.empty:
        missing_counts = count_missing(df)
        report_missing(missing_counts)
    else:
        print("No data to analyze for missing values.")
