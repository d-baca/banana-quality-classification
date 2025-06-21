import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.exploratory_data_analysis.load_csv_data import load_csv_data


def plot_correlation_matrix(df, out_path="reports/figures/correlation_matrix.png"):
    """
    Tworzy macierz korelacji dla wszystkich kolumn numerycznych w DataFrame i zapisuje ją jako obraz.
    """
    if df.empty:
        print("DataFrame is empty. No correlation matrix to plot.")
        return

    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True,
                fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Correlation Matrix")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Correlation matrix saved to {out_path}")


if __name__ == "__main__":
    df = load_csv_data()
    if not df.empty:
        plot_correlation_matrix(df)
    else:
        print("No data to plot correlation matrix.")
