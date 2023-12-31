import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def get_auc(model, X, y, out_features: int):
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch in range(len(X)//32): # Need to fix this
            print(f"Prediction batch: {batch}")
            inputs = X[batch*32:batch*32+32]
            outputs = model(inputs)
            predictions.append(outputs)

    print("Predictions done")
    print(len(predictions))
    predictions_tensor = torch.cat(predictions, dim=0)
    
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(predictions_tensor).numpy()
    y_np = np.array(y)

    # Make sure y_true has at least 2 classes present!!!

    # Calculate AUC for each class and average
    auc_scores = []
    for i in range(out_features):  # For each class
        binary_label = (y_np == i).astype(int)
        class_prob = probabilities[:, i]
        auc = roc_auc_score(binary_label, class_prob)
        auc_scores.append(auc)

    average_auc = np.mean(auc_scores)

    return average_auc