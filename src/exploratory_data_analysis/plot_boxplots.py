import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.exploratory_data_analysis.load_csv_data import load_csv_data


def plot_boxplots(df, out_dir="reports/figures/boxplots.png"):
    """
    Tworzy wykresy pudełkowe dla wszystkich kolumn numerycznych w DataFrame i zapisuje je jako obraz.
    """
    if df.empty:
        print("DataFrame is empty. No boxplots to plot.")
        return

    os.makedirs(out_dir, exist_ok=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.xlabel(col)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{col}_boxplot.png"))
        plt.close()
        print(f"Boxplot for {col} saved to {out_dir}/{col}_boxplot.png")


if __name__ == "__main__":
    df = load_csv_data()
    if not df.empty:
        plot_boxplots(df)
    else:
        print("No data to plot boxplots.")
