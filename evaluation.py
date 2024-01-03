import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def get_auc(model, X, y, out_features: int):
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

    # Calculate AUC for each class and average
    auc_scores = []
    for i in range(out_features):  # For each class
        binary_label = (y_np == i).astype(int)
        class_prob = probabilities[:, i]
        auc = roc_auc_score(binary_label, class_prob)
        auc_scores.append(auc)

    average_auc = np.mean(auc_scores)

    return average_auc