import math
import numpy as np
import torch
from torch import Tensor


def sample_gaussian_mixture_discrete(n_samples, means, covs, grid_size=256, xlim=(-6,6), ylim=(-6,6)):
    """
    Sample from a mixture of 2D Gaussians and then discretize into a grid.

    Returns:
      grid_idxs: array of shape (n_samples, 2), each row (i,j) in [0,grid_size-1]^2
    """
    # 1) Draw continuous samples
    n_gaussians = len(means)
    samples = np.empty((n_samples, 2))
    for k in range(n_samples):
        idx = np.random.randint(0, n_gaussians)
        samples[k] = np.random.multivariate_normal(means[idx], covs[idx])

    # 2) Define the edges of the grid
    x_edges = np.linspace(xlim[0], xlim[1], grid_size+1)
    y_edges = np.linspace(ylim[0], ylim[1], grid_size+1)

    # 3) Digitize each coordinate
    #    np.digitize returns indices in 1..grid_size, so we subtract 1 to get 0..grid_size-1
    i = np.digitize(samples[:,0], bins=x_edges) - 1
    j = np.digitize(samples[:,1], bins=y_edges) - 1

    # 4) Clip any samples that fell exactly on the rightmost/top edge
    i = np.clip(i, 0, grid_size-1)
    j = np.clip(j, 0, grid_size-1)

    return np.stack([i,j], axis=1)
  

means_1 = [
    (-4,  4), (0,  4), (4,  4),
    (-4,  0), (0,  0), (4,  0),
    (-4, -4), (0, -4), (4, -4),
]
means_2 = [
    (-4, 0), (-2*math.sqrt(2), 2*math.sqrt(2)), (0, 4), (2*math.sqrt(2), 2*math.sqrt(2)),
    (4, 0), (2*math.sqrt(2), -2*math.sqrt(2)), (0, -4), (-2*math.sqrt(2), -2*math.sqrt(2)),
]


def generate_samples(type:str, n_grid_points: int = 128, batch_size: int = 200, device: str = "cpu") -> Tensor:
    if type == "1":
        means = means_1
    elif type == "2":
        means = means_2
    else:
        raise ValueError("type must be either '1' or '2'")
    
    covs = [np.array([[0.3, 0],[0,0.3]]) for _ in means]
    
    grid_samples = sample_gaussian_mixture_discrete(batch_size, means, covs, grid_size=n_grid_points)
    return torch.LongTensor(grid_samples).to(device)
