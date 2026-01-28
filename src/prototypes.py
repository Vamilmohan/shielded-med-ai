import torch
import numpy as np

def build_class_prototypes(model, dataloader, num_classes):
    model.eval()
    features = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for x, y in dataloader:
            f = model.forward_features(x)
            for i in range(len(y)):
                features[y[i]].append(f[i].cpu().numpy())

    prototypes = {}
    for c in range(num_classes):
        prototypes[c] = np.mean(features[c], axis=0)

    return prototypes
