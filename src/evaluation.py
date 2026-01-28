import torch
import numpy as np
from sklearn.metrics import roc_curve, auc

def compute_roc_auc(model, dataloader):
    model.eval()
    y_true = []
    y_scores = []

    with torch.no_grad():
        for x, y in dataloader:
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            y_true.extend(y.numpy())
            y_scores.extend(probs.numpy())

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc
