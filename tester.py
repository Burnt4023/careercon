import torch
import joblib
import numpy as np
import pandas as pd
from DataSet import CareerConDataset, MLPClassifier
from sklearn.svm import SVC


def load_and_scale_X_test(X_path, scaler):
    X_df = pd.read_csv(X_path)
    sensor_cols = [c for c in X_df.columns if c not in ['row_id', 'series_id', 'measurement_number']]
    vectors = []
    series_ids = []

    for series_id, grp in X_df.groupby('series_id', sort=True):
        data = grp[sensor_cols].values  # (128, n_feats)
        data_norm = scaler.transform(data)
        vector = data_norm.reshape(-1).astype(np.float32)
        vectors.append(vector)
        series_ids.append(series_id)

    X_test = np.stack(vectors, axis=0)
    return series_ids, X_test


def predict_mlp(model_path, X_test, input_dim, num_classes, class_mapping):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPClassifier(input_dim=input_dim, num_classes=num_classes, hidden_dims=[128])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        outputs = model(X_test_tensor)
        preds = outputs.argmax(dim=1).cpu().numpy()

    idx_to_label = dict(class_mapping)
    predicted_labels = [idx_to_label[p] for p in preds]
    return predicted_labels


def predict_svm(model_path, X_test, class_mapping):
    # Cargar modelo SVM
    svm_model = joblib.load(model_path)
    preds = svm_model.predict(X_test)

    idx_to_label = dict(class_mapping)
    predicted_labels = [idx_to_label[p] for p in preds]
    return predicted_labels


def main():
    X_train_path = 'data/X_train.csv'
    y_train_path = 'data/y_train.csv'
    X_test_path = 'data/X_test.csv'

    scaler = joblib.load('scaler.pkl')

    dataset = CareerConDataset(X_train_path, y_train_path, scaler=scaler)
    class_mapping = dataset.class_mapping  # lista de tuplas (idx, label)
    num_classes = dataset.num_classes

    series_ids, X_test = load_and_scale_X_test(X_test_path, scaler)
    input_dim = X_test.shape[1]

    # Predicción MLP
    predicted_labels_mlp = predict_mlp('best_mlp_model.pt', X_test, input_dim, num_classes, class_mapping)

    # Guardar CSV resultados MLP
    df_mlp = pd.DataFrame({'series_id': series_ids, 'surface': predicted_labels_mlp})
    df_mlp.to_csv('predicciones_mlp.csv', index=False)
    print("Predicciones MLP guardadas en 'predicciones_mlp.csv'")

    # Predicción SVM
    predicted_labels_svm = predict_svm('best_svm_model.pkl', X_test, class_mapping)

    # Guardar CSV resultados SVM
    df_svm = pd.DataFrame({'series_id': series_ids, 'surface': predicted_labels_svm})
    df_svm.to_csv('predicciones_svm.csv', index=False)
    print("Predicciones SVM guardadas en 'predicciones_svm.csv'")


if __name__ == '__main__':
    main()