from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from DataSet import CareerConDataset
from joblib import dump


def main():
    X_path = 'data/X_train.csv'
    y_path = 'data/y_train.csv'

    valid_split = 0.2
    random_state = 42

    print("==> Cargando y preparando datos…")
    dataset = CareerConDataset(X_path=X_path, y_path=y_path, scaler=None)
    dump(dataset.scaler, 'scaler.pkl')

    X_all = dataset.vectors
    y_all = dataset.labels

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=valid_split, stratify=y_all, random_state=random_state
    )

    print("==> Entrenando Random Forest…")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )

    rf_model.fit(X_train, y_train)

    # Validación
    val_preds = rf_model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)

    print(f"Random Forest Val Acc: {val_acc:.4f}")

    # Guardar modelo
    dump(rf_model, 'best_rf_model.pkl')
    print("Modelo Random Forest guardado en 'best_rf_model.pkl'")


if __name__ == '__main__':
    main()