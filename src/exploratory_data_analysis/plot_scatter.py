import os
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.exploratory_data_analysis.load_csv_data import load_csv_data


def plot_pairwise_scatter(df, out_dir="reports/figures/pairwise_scatter"):
    """
    Tworzy wykresy rozrzutu dla wszystkich par kolumn numerycznych w DataFrame i zapisuje je jako obraz.
    """
    if df.empty:
        print("DataFrame is empty. No scatter plots to plot.")
        return

    os.makedirs(out_dir, exist_ok=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        print("Not enough numeric columns to create pairwise scatter plots.")
        return

    pairwise_combinations = itertools.combinations(numeric_cols, 2)

    for col1, col2 in pairwise_combinations:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=df, x=col1, y=col2, hue='Quality', alpha=0.6)
        plt.title(f"Scatter plot of {col1} vs {col2}")
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{col1}_vs_{col2}_scatter.png"))
        plt.close()
        print(
            f"Scatter plot for {col1} vs {col2} saved to {out_dir}/{col1}_vs_{col2}_scatter.png")


if __name__ == "__main__":
    df = load_csv_data()
    if not df.empty:
        plot_pairwise_scatter(df)
    else:
        print("No data to plot pairwise scatter plots.")
