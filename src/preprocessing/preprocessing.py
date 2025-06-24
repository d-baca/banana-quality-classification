import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from pathlib import Path
import json
from src.exploratory_data_analysis.load_csv_data import load_csv_data


class Preprocessor:
    """
    Pipeline preprocessing dla datasetu Banana Quality. Kroki obejmują:
    1. Wczytanie danych z pliku CSV.
    2. Imputacja brakujących wartości (bez akcji bo nie ma braków).
    3. Obsługa wartości odstających (outlier_method="remove" -> detekcja (Z-score & IQR) + usunięcie wierszy z >=2 outlierami,
                                     outlier_method="winsor" -> winsoryzacja 1% ogonów,
                                     outlier_method="log"    -> log1p-transformacja po przesunięciu do >=0).
    4. Feature engineering.
    5. Skalowanie cech (StandardScaler, MinMaxScaler, RobustScaler).
    6. Selekcja cech (L1, DT, RFE).
    7. Zapis przetworzonych danych do CSV + JSON (transakcje).
    """

    def __init__(self, outlier_method: str = "remove", scaler_method: str = "standard", output_dir: str = "data/preprocessed"):
        self.outlier_method = outlier_method  # "remove", "winsor", "log"
        self.scaler_method = scaler_method  # "standard", "minmax", "robust"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, file_path: str = None) -> pd.DataFrame:
        return load_csv_data(file_path) if file_path else load_csv_data()

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        # Imputacja brakujących wartości (brak braków w tym zbiorze)
        return df.copy()

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Usunięcie rekordów z >=2 wartości odstających i winsoryzacja pozostałych."""
        df_clean = df.copy()
        nums = df_clean.select_dtypes(include=[np.number])

        if self.outlier_method == "remove":
            # Z-score outliery
            z = np.abs(stats.zscore(nums, nan_policy="omit")) > 3
            z_mask = z.sum(axis=1) < 2   # wiersze z <2 outlierami

            # IQR outliery
            Q1 = nums.quantile(0.25)
            Q3 = nums.quantile(0.75)
            IQR = Q3 - Q1
            iq = (nums < (Q1 - 1.5 * IQR)) | (nums > (Q3 + 1.5 * IQR))
            iq_mask = iq.sum(axis=1) < 2  # wiersze z <2 outlierami

            # Połączenie
            mask_keep = z_mask & iq_mask
            df_clean = df_clean.loc[mask_keep].reset_index(drop=True)

        elif self.outlier_method == "winsor":
            # Winsoryzacja kolumn numerycznych z 1% ogonami
            for col in nums.columns:
                arr = winsorize(df_clean[col], limits=(0.01, 0.01))
                df_clean[col] = arr.filled(arr.data)  # dtype float64

        elif self.outlier_method == "log":
            # Przesunięcie logarytmiczne do >=0 i logarytmizacja log1p
            for col in nums.columns:
                col_min = df_clean[col].min()
                shift = 0 if col_min >= 0 else -col_min
                df_clean[col] = np.log1p(df_clean[col] + shift)

        return df_clean

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dodanie nowych cech na podstawie istniejących."""
        df_fe = df.copy()
        eps = 1e-6

        # Dodanie cech pochodnych (stosunki)
        df_fe["weight_to_size"] = df_fe.Weight/(df_fe.Size+eps)
        df_fe["ripeness_to_harvest"] = df_fe.Ripeness/(df_fe.HarvestTime+eps)
        df_fe["acidity_times_softness"] = df_fe.Acidity*df_fe.Softness

        # Potęgi
        for c in ["Size", "Weight", "Sweetness", "Softness", "HarvestTime", "Ripeness", "Acidity"]:
            df_fe[f"{c}^2"] = df_fe[c]**2
            df_fe[f"{c}^3"] = df_fe[c]**3

        # Interakcje między cechami
        df_fe["Size*Weight"] = df_fe.Size*df_fe.Weight
        df_fe["Ripeness*Sweetness"] = df_fe.Ripeness*df_fe.Sweetness
        df_fe["HarvestTime*Acidity"] = df_fe.HarvestTime*df_fe.Acidity

        # Dyskretyzacja binarna cech numerycznych do reguł asocjacyjnych
        for col in ["Ripeness", "Acidity"]:
            df_fe[f"{col.lower()}_bin"] = pd.cut(
                df_fe[col],
                bins=[-np.inf, 0, np.inf],
                labels=["low", "high"],
                include_lowest=True
            )
        return df_fe

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Skalowanie cech w zależności od wybranej metody."""
        nums = df.select_dtypes(include=[np.number]).columns
        variances = df[nums].var()
        non_const = variances[variances > 0].index

        if len(non_const) == 0:
            print("All numeric features are constant. No scaling applied.")
            return df.copy()

        data = df[non_const].values
        if self.scaler_method == "minmax":
            scaler = MinMaxScaler()
        elif self.scaler_method == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        scaled = scaler.fit_transform(data)

        pd.DataFrame(scaler.scale_, index=non_const, columns=["scale"]).to_csv(
            self.output_dir/"scaler_params.csv")

        if hasattr(scaler, "mean_"):
            # StandardScaler
            pd.DataFrame(scaler.mean_, index=non_const, columns=["mean"]) \
              .to_csv(self.output_dir/"scaler_mean.csv")
        elif hasattr(scaler, "min_"):
            # MinMaxScaler
            pd.DataFrame(scaler.min_, index=non_const, columns=["min"]) \
              .to_csv(self.output_dir/"scaler_min.csv")
        elif hasattr(scaler, "center_"):
            # RobustScaler
            pd.DataFrame(scaler.center_, index=non_const, columns=["center"]) \
              .to_csv(self.output_dir/"scaler_center.csv")

        # Odtworzenie DataFrame tylko z nieliniowymi cechami
        df_scaled = pd.DataFrame(scaled, columns=non_const)
        # dodanie z powrotem Quality (i ewentualne inne nie-numeryczne)
        for col in df.columns:
            if col not in non_const:
                df_scaled[col] = df[col].values

        return df_scaled

    def select_features(self, df: pd.DataFrame) -> list[str]:
        """Selekcja cech przy użyciu różnych metod."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols]
        y = df["Quality"].cat.codes  # 0/1 etykiety

        # Jeżeli nie ma rekordów, następuje przerwanie selekcji
        if X.empty or X.shape[1] == 0:
            print(
                "No numeric features available for selection. Exiting feature selection.")
            return []

        # Usunięcie rekordów z brakami (NaN) przed trenowaniem
        mask = X.notna().all(axis=1)
        X = X[mask]
        y = y[mask]

        if X.empty:
            print("No records available after removing NaNs. Exiting feature selection.")
            return []

        # L1 Logistic
        lr = LogisticRegression(
            penalty="l1", solver="saga", C=1.0, max_iter=5000)
        lr.fit(X, y)
        sel_l1 = X.columns[lr.coef_.ravel() != 0]

        # Decision Tree
        dt = DecisionTreeClassifier(max_depth=5)
        dt.fit(X, y)
        importances = pd.Series(dt.feature_importances_, index=X.columns)
        sel_dt = importances[importances > importances.mean()].index

        # RFE
        svc = LinearSVC(penalty="l2", dual=False, max_iter=5000)
        rfe = RFE(estimator=svc, n_features_to_select=10)
        rfe.fit(X, y)
        sel_rfe = X.columns[rfe.support_]

        # Kombinacja wyników
        combined = set(sel_l1) & set(sel_dt) & set(sel_rfe)
        if len(combined) < 5:
            combined = set(sel_l1) | set(sel_dt) | set(sel_rfe)
        selected = sorted(combined)

        print(f"Original numeric features: {X.shape[1]}")
        print(f"L1 (Lasso) selected: {len(sel_l1)}")
        print(
            f"Tree importance selected: {len(sel_dt)} (threshold = {importances.mean():.4f})")
        print(f"RFE selected: {len(sel_rfe)}")
        print(f"Final combined selected: {len(selected)}")

        # Zapis wybranych cech do pliku
        with open(self.output_dir/"selected_features.txt", "w") as f:
            for feat in selected:
                f.write(feat + "\n")

        return selected

    def export_transactions(self, df: pd.DataFrame):
        """Zapis transakcji dla reguł asocjacyjnych do pliku JSON."""
        trans = df[["ripeness_bin", "acidity_bin"]].astype(str).values.tolist()

        with open(self.output_dir/"transactions.json", "w") as f:
            json.dump(trans, f)

    def run_all(self, input_path: str = None):
        df = self.load_data(input_path)
        if df.empty:
            print("DataFrame is empty. Exiting preprocessing.")
            return

        df = self.impute_missing_values(df)
        df = self.handle_outliers(df)
        if df.empty:
            print("DataFrame is empty after outlier handling. Exiting preprocessing.")
            return

        df_fe = self.feature_engineering(df)
        if df_fe.empty:
            print("DataFrame is empty after feature engineering. Exiting preprocessing.")
            return

        df_scaled = self.scale_features(df_fe)
        if df_scaled.empty:
            print("DataFrame is empty after scaling. Exiting preprocessing.")
            return

        self.select_features(df_scaled)
        self.export_transactions(df_fe)

        # Zapis preprocessed
        df_scaled.to_csv(
            self.output_dir/"banana_preprocessed.csv", index=False)
        print("Preprocessing complete. Outputs saved to", self.output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Preprocess Banana Quality data")
    parser.add_argument("--input", default="data/banana_quality.csv")
    parser.add_argument("--outdir", default="data/processed")
    parser.add_argument("--outlier-method",
                        choices=["remove", "winsor", "log"], default="remove")
    parser.add_argument(
        "--scaler", choices=["standard", "minmax", "robust"], default="standard")

    args = parser.parse_args()
    pre = Preprocessor(outlier_method=args.outlier_method,
                       scaler_method=args.scaler,
                       output_dir=args.outdir)
    pre.run_all(args.input)
    print("Preprocessing complete. Outputs saved to", args.outdir)
