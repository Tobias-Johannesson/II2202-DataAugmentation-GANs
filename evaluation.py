import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score

def get_metrics(model, X, y, out_features: int):
    model.eval()
    predictions = []

    # Calculate the number of full batches and the remainder
    full_batches = len(X) // 32
    remainder = len(X) % 32

    with torch.no_grad():
        for batch in range(full_batches):
            print(f"Prediction batch: {batch}")
            inputs = X[batch * 32: (batch + 1) * 32]
            outputs = model(inputs)
            predictions.append(outputs)
        
        # Handle the last batch if there is a remainder
        if remainder:
            inputs = X[-remainder:]
            outputs = model(inputs)
            predictions.append(outputs)

    print("Predictions done")
    predictions_tensor = torch.cat(predictions, dim=0)
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(predictions_tensor).numpy()
    y_np = np.array(y)

    # Calculate metrics for each class and average
    auc_scores = []
    accuracy_scores = []
    f1_scores = []
    recall_scores = []
    for i in range(out_features):  # For each class
        binary_label = (y_np == i).astype(int)
        class_prob = probabilities[:, i]
        pred_label = np.argmax(probabilities, axis=1)

        # Check if there are at least two classes present
        if len(np.unique(binary_label)) > 1:
            auc = roc_auc_score(binary_label, class_prob)
            auc_scores.append(auc)
        else:
            print(f"Skipping AUC for class {i} as it has only one class present in y_true.")
            # You might choose to append a default value or skip it
            # auc_scores.append(None)  # For example

        # Calculate other metrics
        accuracy = accuracy_score(binary_label, pred_label == i)
        f1 = f1_score(binary_label, pred_label == i, zero_division=0)
        recall = recall_score(binary_label, pred_label == i, zero_division=0)

        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        recall_scores.append(recall)

    average_auc = np.mean(auc_scores)
    average_accuracy = np.mean(accuracy_scores)
    average_f1 = np.mean(f1_scores)
    average_recall = np.mean(recall_scores)

    return {
        "average_auc": average_auc,
        "average_accuracy": average_accuracy,
        "average_f1": average_f1,
        "average_recall": average_recall
    }