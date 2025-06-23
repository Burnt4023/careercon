import torch
import joblib
import numpy as np
import pandas as pd
from DataSet import CareerConDataset, MLPClassifier


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

def predict_mlp(model_path, X_test, input_dim, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPClassifier(input_dim=input_dim, num_classes=num_classes, hidden_dims=[128])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        outputs = model(X_test_tensor)
        preds = outputs.argmax(dim=1).cpu().numpy()
    return preds

def main():
    X_train_path = 'data/X_train.csv'
    y_train_path = 'data/y_train.csv'
    X_test_path = 'data/X_test.csv'

    # Cargar scaler
    scaler = joblib.load('scaler.pkl')

    # Cargar dataset de entrenamiento solo para acceder a class_names y num_classes
    dataset = CareerConDataset(X_train_path, y_train_path, scaler=scaler)
    class_names = dataset.class_names
    num_classes = dataset.num_classes

    # Cargar test set con scaler ya entrenado
    series_ids, X_test = load_and_scale_X_test(X_test_path, scaler)
    input_dim = X_test.shape[1]

    # Predicciones con MLP
    preds = predict_mlp('best_mlp_model.pt', X_test, input_dim, num_classes)

    # Mapear predicciones a etiquetas string usando dataset.class_names
    idx_to_label = dict(dataset.class_mapping)

    predicted_labels = [idx_to_label[p] for p in preds]

    # Guardar CSV
    df_out = pd.DataFrame({'series_id': series_ids, 'predicted_label': predicted_labels})
    df_out.to_csv('predicciones.csv', index=False)
    print("Predicciones guardadas en 'predicciones.csv'")


if __name__ == '__main__':
    main()
