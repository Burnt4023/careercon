import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from DataSet import CareerConDataset
import numpy as np
from joblib import dump

class CNNClassifier(nn.Module):
    def __init__(self, input_length, num_classes):
        super().__init__()
        # input_length es la dimensión del vector plano, lo convertiremos a (canal=1, largo=input_length)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)

        # Calcula output_length tras poolings (2 poolings de factor 2)
        out_len = input_length // 4

        self.fc1 = nn.Linear(64 * out_len, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch, input_length)
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

def train_cnn():
    X_path = 'data/X_train.csv'
    y_path = 'data/y_train.csv'

    valid_split = 0.2
    random_state = 42
    num_epochs = 25
    batch_size = 128
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("==> Cargando y preparando datos…")
    dataset = CareerConDataset(X_path=X_path, y_path=y_path, scaler=None)
    dump(dataset.scaler, 'scaler.pkl')

    X_all = dataset.vectors
    y_all = dataset.labels
    num_classes = dataset.num_classes
    input_dim = X_all.shape[1]

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=valid_split, stratify=y_all, random_state=random_state
    )

    model = CNNClassifier(input_length=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        permutation = np.random.permutation(len(X_train))

        for i in range(0, len(X_train), batch_size):
            idx = permutation[i:i + batch_size]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val).argmax(dim=1)
            val_acc = accuracy_score(y_val.cpu(), val_preds.cpu())

        print(f"Epoch {epoch:02d}/{num_epochs} ▸ Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_cnn_model.pt')

    print(f"\nMejor accuracy en validación CNN: {best_val_acc:.4f}")


if __name__ == '__main__':
    train_cnn()