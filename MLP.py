from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from DataSet import CareerConDataset, MLPClassifier
import torch
import torch.nn as nn
import numpy as np
from joblib import dump


def main():
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

    model = MLPClassifier(input_dim=input_dim, hidden_dims=[128], num_classes=num_classes, dropout=0.5).to(device)
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
            torch.save(model.state_dict(), 'best_mlp_model.pt')

    print(f"\nMejor accuracy en validación: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()