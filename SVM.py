from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from DataSet import CareerConDataset
from joblib import dump
import numpy as np

def main():
    X_path = 'data/X_train.csv'
    y_path = 'data/y_train.csv'

    valid_split = 0.2
    random_state = 42
    num_epochs = 25  # Para simular epochs y mostrar progreso (aunque SVM no usa epochs)

    print("==> Preparando datos de entrenamiento y validación…")
    full_dataset = CareerConDataset(X_path=X_path, y_path=y_path, scaler=None)
    dump(full_dataset.scaler, 'scaler.pkl')

    X_all = full_dataset.vectors
    y_all = full_dataset.labels

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=valid_split, stratify=y_all, random_state=random_state
    )

    print("==> Iniciando entrenamiento de SVM…")
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=random_state)

    # Entrenar SVM (una sola vez, no epochs)
    svm_model.fit(X_train, y_train)

    # Para simular epochs y mostrar la misma estructura que en MLP
    for epoch in range(1, num_epochs + 1):
        # Predecimos en validación y calculamos accuracy
        val_preds = svm_model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        print(f"Epoch {epoch:02d}/{num_epochs} ▸ Val Acc: {val_acc:.4f}")

    # Guardar modelo SVM entrenado
    dump(svm_model, 'best_svm_model.pkl')

    print(f"\nMejor accuracy en validación (SVM): {val_acc:.4f}")
    print("Entrenamiento finalizado. Modelo SVM guardado en 'best_svm_model.pkl'.\n")

if __name__ == '__main__':
    main()
