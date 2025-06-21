import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.exploratory_data_analysis.load_csv_data import load_csv_data


def plot_histograms(df, out_dir="reports/figures/histograms", bins=30):
    """
    Tworzy histogramy dla wszystkich kolumn numerycznych w DataFrame.
    """
    if df.empty:
        print("DataFrame is empty. No histograms to plot.")
        return

    os.makedirs(out_dir, exist_ok=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        # kde oznacza Kernel Density Estimate, czyli estymację gęstości rozkładu
        sns.histplot(df[col], bins=bins, kde=True)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{col}_histogram.png"))
        plt.close()
        print(f"Histogram for {col} saved to {out_dir}/{col}_histogram.png")


if __name__ == "__main__":
    df = load_csv_data()
    if not df.empty:
        plot_histograms(df)
    else:
        print("No data to plot histograms.")
