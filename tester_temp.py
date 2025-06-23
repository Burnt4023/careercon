import torch
import joblib
import numpy as np
import pandas as pd
from DataSet import CareerConDataset, MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import torch.nn.functional as F
import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self, input_length, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)

        out_len = input_length // 4

        self.fc1 = nn.Linear(64 * out_len, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, input_length)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def predict_cnn(model_path, X_test, input_dim, num_classes, class_mapping):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNClassifier(input_length=input_dim, num_classes=num_classes)
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
    svm_model = joblib.load(model_path)
    preds = svm_model.predict(X_test)

    idx_to_label = dict(class_mapping)
    predicted_labels = [idx_to_label[p] for p in preds]
    return predicted_labels


def predict_rf(model_path, X_test, class_mapping):
    rf_model = joblib.load(model_path)
    preds = rf_model.predict(X_test)

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
    df_mlp = pd.DataFrame({'series_id': series_ids, 'surface': predicted_labels_mlp})
    df_mlp.to_csv('predicciones_mlp.csv', index=False)
    print("Predicciones MLP guardadas en 'predicciones_mlp.csv'")

    # Predicción SVM
    predicted_labels_svm = predict_svm('best_svm_model.pkl', X_test, class_mapping)
    df_svm = pd.DataFrame({'series_id': series_ids, 'surface': predicted_labels_svm})
    df_svm.to_csv('predicciones_svm.csv', index=False)
    print("Predicciones SVM guardadas en 'predicciones_svm.csv'")

    # Predicción Random Forest
    predicted_labels_rf = predict_rf('best_rf_model.pkl', X_test, class_mapping)
    df_rf = pd.DataFrame({'series_id': series_ids, 'surface': predicted_labels_rf})
    df_rf.to_csv('predicciones_rf.csv', index=False)
    print("Predicciones Random Forest guardadas en 'predicciones_rf.csv'")

    # Predicción CNN
    predicted_labels_cnn = predict_cnn('best_cnn_model.pt', X_test, input_dim, num_classes, class_mapping)
    df_cnn = pd.DataFrame({'series_id': series_ids, 'surface': predicted_labels_cnn})
    df_cnn.to_csv('predicciones_cnn.csv', index=False)
    print("Predicciones CNN guardadas en 'predicciones_cnn.csv'")


if __name__ == '__main__':
    main()