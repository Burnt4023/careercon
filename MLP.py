from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from DataSet import CareerConDataset, MLPClassifier
import torch
import torch.nn as nn
import numpy as np

def main():
    X_path = 'data/X_train.csv'
    y_path = 'data/y_train.csv'

    valid_split = 0.2
    random_state = 42
    num_epochs = 25
    batch_size = 128
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("==> Preparando datos de entrenamiento y validación…")
    full_dataset = CareerConDataset(X_path=X_path, y_path=y_path, scaler=None)
    num_classes = full_dataset.num_classes

    X_all = full_dataset.vectors
    y_all = full_dataset.labels

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=valid_split, stratify=y_all, random_state=random_state
    )

    input_dim = X_train.shape[1]

    print("==> Iniciando entrenamiento de MLP…")
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=[128],
        num_classes=num_classes,
        dropout=0.5
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_np = y_val  # numpy array para metric

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        permutation = np.random.permutation(len(X_train))

        for i in range(0, len(X_train), batch_size):
            indices = permutation[i:i + batch_size]
            xb = X_train[indices].to(device)
            yb = y_train[indices].to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

        # Validación
        model.eval()
        with torch.no_grad():
            train_preds = model(X_train.to(device)).argmax(dim=1).cpu().numpy()
            val_preds = model(X_val).argmax(dim=1).cpu().numpy()

        train_acc = accuracy_score(y_train.numpy(), train_preds)
        val_acc = accuracy_score(y_val_np, val_preds)

        print(f"Epoch {epoch:02d}/{num_epochs} ▸ Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
    print(f"\nMejor accuracy en validación (MLP): {best_val_acc:.4f}")
    
    return model

if __name__ == '__main__':
    main()