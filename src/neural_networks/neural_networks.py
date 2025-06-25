import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, RocCurveDisplay
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def load_data(data_path, features_file=None):
    """Załadowanie danych z pliku CSV i opcjonalnie wybranych cech."""
    df = pd.read_csv(data_path)
    if features_file:
        with open(features_file) as f:
            features = [l.strip() for l in f if l.strip()]
        X = df[features].values
    else:
        X = df.drop(columns=['Quality', 'ripeness_bin', 'acidity_bin']).values
    y = df['Quality'].astype('category').cat.codes.values
    return X, y


def build_mlp(input_dim, layers_config, activation, optimizer):
    """Tworzenie modelu Multi-layer Perceptron z Dropout i konfiguracją warstw."""
    model = models.Sequential()
    for i, units in enumerate(layers_config):
        if i == 0:
            model.add(layers.Dense(units, activation=activation,
                      input_shape=(input_dim,)))
        else:
            model.add(layers.Dense(units, activation=activation))

        # Dodanie Dropout po każdej warstwie, aby ograniczyć przeuczenie
        model.add(layers.Dropout(0.2))

    # Warstwa wyjściowa
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


def build_autoencoder(input_dim, encoding_dim):
    """Tworzenie autoenkodera z podanymi parametrami."""
    # Enkoder i dekoder dla autoenkodera
    inp = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation='relu')(inp)
    decoded = layers.Dense(input_dim, activation='linear')(encoded)
    auto = models.Model(inp, decoded)
    auto.compile(optimizer='adam', loss='mse')

    # Separacja modelu enkodera
    encoder = models.Model(inp, encoded)
    return auto, encoder


def evaluate_mlp(X_train, y_train, X_val, y_val, config, output_dir):
    """Trenowanie i ewaluacja modelu MLP na danych treningowych i walidacyjnych."""
    base_name = f"{'x'.join(map(str,config['layers']))}_{config['activation']}_{config['optimizer']}_bs{config['batch_size']}"
    base_dir = os.path.join(output_dir, base_name)
    ep_dir = os.path.join(base_dir, f"ep{config['epochs']}")
    os.makedirs(ep_dir, exist_ok=True)

    model = build_mlp(
        input_dim=X_train.shape[1],
        layers_config=config['layers'],
        activation=config['activation'],
        optimizer=config['optimizer']
    )

    # Callbacki - EarlyStopping i ReduceLROnPlateau dla efektywnego treningu
    cb_es = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    cb_rl = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=2)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=[cb_es, cb_rl],
        verbose=2
    )

    # Predykcja i obliczenie metryk
    y_prob = model.predict(X_val).reshape(-1)
    y_pred = (y_prob > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_prob)
    }

    # Raport klasyfikacji
    report = classification_report(y_val, y_pred, target_names=['Bad', 'Good'])
    with open(os.path.join(ep_dir, f"classification_report.txt"), 'w') as f:
        f.write(report)

    # Zapisanie metryk i historii treningu do pliku
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(ep_dir, "mlp_history.csv"), index=False)

    # Krzywa LOSS + krzywa ACCURACY
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history.history['loss'],  label='train_loss')
    axes[0].plot(history.history['val_loss'], label='val_loss')
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Accuracy
    axes[1].plot(history.history['accuracy'],     label='train_acc')
    axes[1].plot(history.history['val_accuracy'], label='val_acc')
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.suptitle(config['name'], fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(ep_dir, "loss_acc.png"))
    plt.close(fig)

    # Krzywa AUC + krzywa ROC
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Krzywa uczenia AUC
    axes[0].plot(history.history['auc'],     label='train_auc')
    axes[0].plot(history.history['val_auc'], label='val_auc')
    axes[0].set_title("AUC")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Krzywa ROC
    RocCurveDisplay.from_predictions(y_val, y_prob, ax=axes[1])
    axes[1].set_title("ROC Curve")
    fig.suptitle(config['name'], fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(ep_dir, "auc_roc.png"))
    plt.close(fig)

    return metrics


def main():
    parser = argparse.ArgumentParser("MLP & Autoencoder")
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--features-file', default=None)
    parser.add_argument('--output-dir', default='neural_networks_results')
    parser.add_argument('--autoencoder', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Załadowanie danych
    X, y = load_data(args.data_path, args.features_file)

    # Podział danych na zbiór treningowy i walidacyjny (80% train, 20% val)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Normalizacja danych
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    # Konfiguracja i trenowanie modeli MLP
    layer_options = [[64], [128], [64, 32]]
    activations = ['relu', 'tanh']
    optimizers = ['adam']
    batch_sizes = [32, 64]
    epochs_options = [50, 100]

    results = {}
    for layers_cfg in layer_options:
        for act in activations:
            for opt in optimizers:
                for bs in batch_sizes:
                    for ep in epochs_options:
                        name = f"{'x'.join(map(str,layers_cfg))}_{act}_{opt}_bs{bs}_ep{ep}"
                        cfg = {'layers': layers_cfg, 'activation': act,
                               'optimizer': opt, 'batch_size': bs, 'epochs': ep, 'name': name}
                        print("Training MLP", name)
                        m = evaluate_mlp(X_train, y_train, X_val,
                                         y_val, cfg, args.output_dir)
                        results[name] = m

    # Zapisanie wyników do pliku CSV
    df = pd.DataFrame(results).T
    df.to_csv(os.path.join(args.output_dir, 'mlp_summary.csv'))

    # Wykres porównania TOP10 konfiguracji MLP według ACCURACY
    top_acc = df.sort_values('accuracy', ascending=False).head(10)
    plt.figure(figsize=(8, 5))
    top_acc['accuracy'].plot(kind='barh')
    plt.title("Top 10 MLP configs by Accuracy")
    plt.xlabel("Accuracy")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'top10_accuracy.png'))
    plt.close()

    # Wykres porównania TOP10 konfiguracji MLP według ROC-AUC
    top_auc = df.sort_values('roc_auc', ascending=False).head(10)
    plt.figure(figsize=(8, 5))
    top_auc['roc_auc'].plot(kind='barh', color='C1')
    plt.title("Top 10 MLP configs by ROC-AUC")
    plt.xlabel("ROC-AUC")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'top10_roc_auc.png'))
    plt.close()

    if args.autoencoder:
        # Autoekoder
        ae, encoder = build_autoencoder(input_dim=X_train.shape[1],
                                        encoding_dim=max(2, X_train.shape[1]//4))

        cb_ae = [callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                 callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)]

        history = ae.fit(X_train, X_train,
                         validation_data=(X_val, X_val),
                         epochs=100, batch_size=64,
                         callbacks=cb_ae,
                         verbose=2)

        # Predykcja i ewaluacja enkodera
        Z_train = encoder.predict(X_train)
        Z_val = encoder.predict(X_val)

        # Trenowanie klasyfikatora na zakodowanych danych
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=500).fit(Z_train, y_train)
        y_pred = clf.predict(Z_val)
        y_proba = clf.predict_proba(Z_val)[:, 1]
        ae_metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_proba)
        }
        pd.DataFrame(history.history).to_csv(os.path.join(
            args.output_dir, 'autoencoder_history.csv'), index=False)
        pd.DataFrame({'metric': list(ae_metrics.keys()), 'value': list(ae_metrics.values())}).to_csv(
            os.path.join(args.output_dir, 'autoencoder_summary.csv'), index=False)

    print("Done. Results in", args.output_dir)


if __name__ == '__main__':
    main()
