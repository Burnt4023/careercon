
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class CareerConDataset(Dataset):
    """
    Lee X_train.csv y y_train.csv:
      - X_train.csv: row_id, series_id, measurement_number, sensor_1…sensor_10
      - y_train.csv: series_id, surface (string)
    """

    def __init__(self, X_path, y_path, scaler=None):
        super().__init__()
        # 1) Cargo etiquetas y factorizo
        y_df = pd.read_csv(y_path)
        cols = list(y_df.columns)
        if cols[0] != 'series_id':
            raise ValueError(f"Se esperaba primera columna 'series_id', pero es '{cols[0]}'")
        label_col = cols[1]

        surfaces = y_df[label_col].values
        labels_factorized, uniques = pd.factorize(surfaces)
        y_df['label_idx'] = labels_factorized
        self.num_classes = len(uniques)

        # 2) Cargo X_train y saco columnas de sensores
        X_df = pd.read_csv(X_path)
        sensor_cols = [c for c in X_df.columns if c not in ['row_id', 'series_id', 'measurement_number']]

        # 3) Merge para insertar label_idx en cada fila de X_df según series_id
        merged = X_df.merge(y_df[['series_id', 'label_idx']], on='series_id', how='left')
        merged.sort_values(['series_id', 'measurement_number'], inplace=True)

        # 4) Ajusto StandardScaler SOBRE TODO X_df (solo una vez):
        if scaler is None:
            all_data = X_df[sensor_cols].values
            self.scaler = StandardScaler().fit(all_data)
        else:
            self.scaler = scaler

        # 5) Agrupo por series_id → construyo self.vectors y self.labels
        vectors = []
        labels = []
        for series_id, grp in tqdm(merged.groupby('series_id', sort=True),
                                   desc="Construyendo dataset", leave=False):
            data = grp[sensor_cols].values  # (128, n_feats)
            label_idx = int(grp['label_idx'].iloc[0])

            data_norm = self.scaler.transform(data)
            vector = data_norm.reshape(-1).astype(np.float32)  # (128*n_feats,)
            vectors.append(vector)
            labels.append(label_idx)

        self.vectors = np.stack(vectors, axis=0)   # (n_series, 128*n_feats)
        self.labels = np.array(labels, dtype=np.int64)  # (n_series,)

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx], self.labels[idx]
    
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], num_classes=1, dropout=0.5):
        """
        input_dim: 128 * n_feats
        hidden_dims: lista de neuronas en capas ocultas
        num_classes: número real de clases (ej. 9)
        """
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch_size, input_dim)
        return self.net(x)