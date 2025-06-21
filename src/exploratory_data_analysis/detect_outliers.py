import pandas as pd
import numpy as np
from scipy import stats
from src.exploratory_data_analysis.load_csv_data import load_csv_data


def zscore_outliers(df, threshold=3.0):
    """
    Identyfikacja obserwacji odstających na podstawie z-score.
    """
    nums = df.select_dtypes(include=[np.number])
    z = np.abs(stats.zscore(nums, nan_policy='omit'))
    # liczba obserwacji odstających dla każdej kolumny
    return pd.Series((z > threshold).sum(axis=0), index=nums.columns)


def iqr_outliers(df):
    """
    Identyfikacja obserwacji odstających na podstawie IQR.
    """
    nums = df.select_dtypes(include=[np.number])
    Q1 = nums.quantile(0.25)
    Q3 = nums.quantile(0.75)
    IQR = Q3 - Q1
    lower = ((nums < (Q1 - 1.5 * IQR))).sum()
    upper = ((nums > (Q3 + 1.5 * IQR))).sum()
    # liczba obserwacji odstających dla każdej kolumny
    return pd.DataFrame({
        "lower": lower,
        "upper": upper
    })


if __name__ == "__main__":
    df = load_csv_data()
    z_counts = zscore_outliers(df)
    iqr_counts = iqr_outliers(df)
    print("Z-score outliers:\n", z_counts)
    print("\nIQR outliers:\n", iqr_counts)
