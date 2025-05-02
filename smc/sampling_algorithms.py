import torch
from torch import Tensor

def stratified_resample(log_weights: Tensor):
    N = log_weights.shape[0]
    weights = torch.exp(log_weights)
    cdf = torch.cumsum(weights, dim=0)

    # Stratified uniform samples
    u = (torch.arange(N, dtype=torch.float32, device=log_weights.device) + torch.rand(N, device=log_weights.device)) / N

    indices = torch.searchsorted(cdf, u, right=True)
    return indices

def systematic_resample(log_weights: Tensor):
    N = log_weights.shape[0]
    weights = torch.exp(log_weights)
    cdf = torch.cumsum(weights, dim=0)

    # Systematic uniform samples
    u0 = torch.rand(1, device=log_weights.device) / N
    u = u0 + torch.arange(N, dtype=torch.float32, device=log_weights.device) / N

    indices = torch.searchsorted(cdf, u, right=True)
    return indices

def multinomial_resample(log_weights: Tensor):
    N = log_weights.shape[0]
    resampled_indices = torch.multinomial(torch.exp(log_weights), N, replacement=True)
    return resampled_indices
