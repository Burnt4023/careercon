from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from DataSet import CareerConDataset


def main():
    X_path = 'data/X_train.csv'
    y_path = 'data/y_train.csv'

    valid_split = 0.2
    random_state = 42

    print("==> Preparando datos de entrenamiento y validación…")
    # Reutilizamos CareerConMLPDataset para cargar y normalizar
    full_dataset = CareerConDataset(X_path=X_path, y_path=y_path, scaler=None)
    num_classes = full_dataset.num_classes

    # Extraemos vectores y etiquetas (NumPy)
    X_all = full_dataset.vectors
    y_all = full_dataset.labels

    # Dividimos en train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=valid_split, stratify=y_all, random_state=random_state
    )

    print("==> Iniciando entrenamiento de SVM…")
    # Creamos el clasificador SVM (kernel RBF por defecto)
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=random_state, probability=True)

    # Entrenamos
    svm_model.fit(X_train, y_train)

    # Predicciones para train y val
    y_pred_train = svm_model.predict(X_train)
    y_pred_val = svm_model.predict(X_val)

    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_val, y_pred_val)

    # Mostramos el mismo estilo de salida que en las redes neuronales
    print(f"\n  ▸ Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    print(f"\nMejor accuracy en validación (SVM): {val_acc:.4f}")
    print("Entrenamiento finalizado. Modelo SVM listo para usarse.\n")
    
    return svm_model

if __name__ == '__main__':
    main()