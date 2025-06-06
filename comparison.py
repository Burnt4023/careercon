import MLP
import SVM
import DataSet
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, recall_score, accuracy_score, precision_score

def compareModels():

    # Obtener modelos
    mlp_model = MLP.main()
    svm_model = SVM.main()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Obtener datos de test
    X_path = 'data/X_train.csv'
    y_path = 'data/y_train.csv'
    full_dataset = DataSet.CareerConDataset(X_path=X_path, y_path=y_path, scaler=None)
    X_test = full_dataset.vectors
    y_test = full_dataset.labels

    # Evaluar MLP
    print("\nEvaluando MLP...")
    mlp_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = mlp_model(X_test_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        y_pred_mlp = logits.argmax(dim=1).cpu().numpy()

    # Evaluar SVM
    print("\nEvaluando SVM...")
    y_pred_svm = svm_model.predict(X_test)
    svm_probs = svm_model.predict_proba(X_test)  # Usa probabilidades

    # Métricas
    print("\nCalculando métricas...")

    # MLP
    f1_mlp = f1_score(y_test, y_pred_mlp, average='weighted')
    roc_auc_mlp = roc_auc_score(y_test, probs, multi_class='ovr')
    recall_mlp = recall_score(y_test, y_pred_mlp, average='weighted')
    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
    precision_mlp = precision_score(y_test, y_pred_mlp, average='weighted')

    # SVM
    f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
    roc_auc_svm = roc_auc_score(y_test, svm_probs, multi_class='ovr')
    recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    precision_svm = precision_score(y_test, y_pred_svm, average='weighted')

    print("---------------------------------------------------")
    print("MLP:")
    print(f"  Accuracy: {accuracy_mlp:.4f}")
    print(f"  Precision: {precision_mlp:.4f}")
    print(f"  Recall: {recall_mlp:.4f}")
    print(f"  F1 Score: {f1_mlp:.4f}")
    print(f"  ROC AUC Score: {roc_auc_mlp:.4f}")
    print("---------------------------------------------------")
    print("SVM:")
    print(f"  Accuracy: {accuracy_svm:.4f}")
    print(f"  Precision: {precision_svm:.4f}")
    print(f"  Recall: {recall_svm:.4f}")
    print(f"  F1 Score: {f1_svm:.4f}")
    print(f"  ROC AUC Score: {roc_auc_svm:.4f}")

if __name__ == '__main__':
    compareModels()