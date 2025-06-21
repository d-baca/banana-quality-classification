from src.exploratory_data_analysis.load_csv_data import load_csv_data
from src.exploratory_data_analysis.describe_data import describe_data, save_description
from src.exploratory_data_analysis.plot_distributions import plot_histograms
from src.exploratory_data_analysis.plot_correlation import plot_correlation_matrix
from src.exploratory_data_analysis.plot_boxplots import plot_boxplots
from src.exploratory_data_analysis.plot_scatter import plot_pairwise_scatter
from src.exploratory_data_analysis.analyze_missing import count_missing, report_missing
from src.exploratory_data_analysis.detect_outliers import zscore_outliers, iqr_outliers


def main():
    # Load data
    df = load_csv_data()

    # 1. Opisowe statystyki i zapis do pliku
    description = describe_data(df)
    save_description(description)
    print("Description saved successfully to reports/figures/descriptive_stats.csv")

    # 2. Histogramy
    plot_histograms(df)
    print("Histograms saved to reports/figures/histograms/")

    # 3. Macierz korelacji
    plot_correlation_matrix(df)
    print("Correlation matrix saved to reports/figures/correlation_matrix.png")

    # 4. Wykresy pudełkowe
    plot_boxplots(df)
    print("Boxplots saved to reports/figures/boxplots/")

    # 5. Wykresy rozrzutu dla par kolumn numerycznych
    plot_pairwise_scatter(df)
    print("Scatter plots saved to reports/figures/scatter/")

    # 6. Sprawdzenie brakujących wartości
    missing_counts = count_missing(df)
    report_missing(missing_counts)
    print("Missing values report saved to reports/figures/missing_report.txt")

    # 7. Wykrywanie wartości odstających
    z_counts = zscore_outliers(df)
    iqr_counts = iqr_outliers(df)
    print("Z-score outliers:\n", z_counts)
    print("\nIQR outliers:\n", iqr_counts)


if __name__ == "__main__":
    main()
