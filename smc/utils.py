import torch

def compute_ess(w, dim=-1):
    ess = (w.sum(dim=dim))**2 / torch.sum(w**2, dim=dim)
    return ess

def compute_ess_from_log_w(log_w, dim=-1):
    return compute_ess(normalize_weights(log_w, dim=dim), dim=dim)

def normalize_weights(log_weights, dim=-1):
    return torch.exp(normalize_log_weights(log_weights, dim=dim))

def normalize_log_weights(log_weights, dim=-1):
    log_weights = log_weights - log_weights.max(dim=dim, keepdims=True)[0]
    log_weights = log_weights - torch.logsumexp(log_weights, dim=dim, keepdims=True)
    return log_weights
