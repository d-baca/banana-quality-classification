import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

transformations = ["remove", "winsor", "log"]

configs = [
    ("standard", StandardScaler()),
    ("minmax",   MinMaxScaler()),
    ("robust",   RobustScaler())
]

# Definicja dwóch klasyfikatorów (modeli)
models = {
    "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=5)
}

results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for transformation in transformations:
    for scaler_name, scaler in configs:
        df = pd.read_csv(
            f"data/processed/{transformation}/{scaler_name}/banana_preprocessed.csv")
        X = df.drop(columns=["Quality", "ripeness_bin", "acidity_bin"])
        y = df["Quality"].astype("category").cat.codes

        for model_name, model in models.items():
            pipe = Pipeline([
                (scaler_name, scaler),
                (model_name, model)
            ])
            cv_scores = cross_validate(
                pipe, X, y,
                cv=cv,
                scoring={"accuracy": "accuracy", "f1": "f1_weighted"},
                return_train_score=False,
                n_jobs=-1
            )
            results.append({
                "transformation": transformation,
                "scaler": scaler_name,
                "model": model_name,
                "accuracy": cv_scores["test_accuracy"].mean(),
                "f1": cv_scores["test_f1"].mean()
            })

df_res = pd.DataFrame(results)
pivot = df_res.pivot_table(index=["transformation", "scaler"], columns="model",
                           values=["accuracy", "f1"])
print(pivot)
