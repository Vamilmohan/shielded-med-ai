import torch

def add_dp_noise(weights, noise_scale=0.02):
    noisy = {}
    for k, v in weights.items():
        noise = torch.normal(0, noise_scale, size=v.shape)
        noisy[k] = v + noise
    return noisy

def federated_average(state_dicts):
    avg = {}
    for key in state_dicts[0]:
        avg[key] = sum(d[key] for d in state_dicts) / len(state_dicts)
    return add_dp_noise(avg)