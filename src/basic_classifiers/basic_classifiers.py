import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, learning_curve
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay)


def load_data(data_path, features_file=None):
    """Załadowanie danych z pliku CSV i ewentualnie wybranie cech."""
    df = pd.read_csv(data_path)
    if features_file:
        with open(features_file, 'r') as f:
            features = [line.strip() for line in f if line.strip()]
        X = df[features]
    else:
        X = df.drop(columns=['Quality', 'ripeness_bin', 'acidity_bin'])
    y = df['Quality'].astype('category').cat.codes
    return X, y


def build_pipelines():
    """Tworzenie różnych klasyfikatorów w postaci pipeline'ów."""
    return {
        "GaussianNB": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GaussianNB())
        ]),
        "BernoulliNB": Pipeline([
            ('binarizer', Binarizer(threshold=0.0)),
            ('classifier', BernoulliNB())
        ]),
        # Dla danych dyskretyzowanych
        'CategoricalNB': Pipeline([
            ('binarizer', Binarizer(threshold=0.0)),
            ('classifier', CategoricalNB())
        ]),
        "kNN": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ]),
        "DecisionTree": Pipeline([
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])
    }


def build_param_grids():
    """Tworzenie słowników z parametrami do GridSearchCV."""
    return {
        "GaussianNB": {
            # Stopień wygładzania wariancji
            'classifier__var_smoothing': [1e-9, 1e-8, 1e-7]
        },
        "BernoulliNB": {
            'classifier__alpha': [0.5, 1.0, 1.5]  # Współczynnik wygładzania
        },
        "CategoricalNB": {
            'classifier__alpha': [0.5, 1.0, 1.5]  # Współczynnik wygładzania
        },
        "kNN": {
            'classifier__n_neighbors': [3, 5, 7, 9],  # Liczba sąsiadów
            # Metryki odległości
            'classifier__metric': ['euclidean', 'manhattan', 'minkowski'],
            'classifier__p': [2, 3]  # Wartość p dla metryki Minkowskiego
        },
        "DecisionTree": {
            'classifier__criterion': ['gini', 'entropy'],  # Kryteria podziału
            # Maksymalna głębokość drzewa
            'classifier__max_depth': [3, 5, 7, 10, None]
        }
    }


def evaluate_model(name, pipeline, params, X, y, output_dir):
    """Ocena modelu przy użyciu GridSearchCV i cross-validation."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline, params,
        cv=cv,
        scoring=['accuracy', 'precision_weighted',
                 'recall_weighted', 'f1_weighted', 'roc_auc'],
        refit='f1_weighted',
        n_jobs=-1
    )
    grid.fit(X, y)
    best_model = grid.best_estimator_

    # Predykcje cross-validation
    y_pred = cross_val_predict(best_model, X, y, cv=cv)
    y_proba = cross_val_predict(
        best_model, X, y, cv=cv, method='predict_proba')[:, 1]

    # Obliczanie metryk
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='weighted'),
        'recall': recall_score(y, y_pred, average='weighted'),
        'f1': f1_score(y, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y, y_proba)
    }
    print(f"\n=== {name} Best Params ===\n{grid.best_params_}")
    print(f"=== {name} CV Metrics ===\n{metrics}")

    # Macierz pomyłek
    disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, normalize='true')
    disp.figure_.suptitle(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_confusion_matrix.png"))
    plt.close()

    # Krzywa ROC
    roc_disp = RocCurveDisplay.from_predictions(y, y_proba)
    roc_disp.figure_.suptitle(f"{name} ROC Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_roc_curve.png"))
    plt.close()

    # Krzywe uczenia
    train_sizes, train_scores, test_scores = learning_curve(
        best_model, X, y, cv=cv, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)
    plt.figure()
    plt.plot(train_sizes, train_mean, 'o-', label='Train Accuracy')
    plt.plot(train_sizes, test_mean,  'o-', label='CV Accuracy')
    plt.title(f"{name} Learning Curve")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_learning_curve.png"))
    plt.close()

    return metrics


def main(args):
    """Główna funkcja do uruchomienia klasyfikatorów."""
    os.makedirs(args.output_dir, exist_ok=True)
    X, y = load_data(args.data_path, args.features_file)
    pipelines = build_pipelines()
    param_grids = build_param_grids()

    results = {}
    for name, pipe in pipelines.items():
        if name not in param_grids:
            continue
        print(f"\n>> Evaluating {name}...")
        metrics = evaluate_model(
            name, pipe, param_grids[name], X, y, args.output_dir)
        results[name] = metrics

    # Zapis wyników do pliku CSV
    summary = pd.DataFrame(results).T
    summary.to_csv(os.path.join(args.output_dir,
                   "basic_classifiers_results_summary.csv"), index=True)
    print("\nAll done! Results saved to:", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple classifiers with grid search and CV")
    parser.add_argument("--data-path",      required=True,
                        help="Path to preprocessed CSV file")
    parser.add_argument("--features-file",  default=None,
                        help="Optional: path to selected_features.txt")
    parser.add_argument(
        "--output-dir",     default="phase3_results", help="Directory to save outputs")
    args = parser.parse_args()
    main(args)
