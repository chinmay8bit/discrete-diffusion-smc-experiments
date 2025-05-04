import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F
from typing import Callable

def plot_smc_results_checkerboard(
    X_0: torch.Tensor,
    W_0: torch.Tensor,
    ess_trace: np.ndarray,
    rewards_trace: np.ndarray,
    particles_trace: list,
    log_weights_trace: list,
    resampling_trace: list,
    num_timesteps: int,
    vocab_size: int,
    num_categories: int,
    compute_rewards_fn: Callable[[torch.Tensor], torch.Tensor],
    interval: int = 100
) -> None:
    """
    Visualize the progression and diagnostics of a Sequential Monte Carlo (SMC) run.

    This function generates several plots:
    1. Scatter plots of particles at different time steps.
    2. Combined plot showing sample diversity and reward variance over time.
    3. Final scatter of initial samples X_0 with computed diversity and average reward.
    4. Combined plot of effective sample size (ESS), sample diversity, and mean rewards over time.

    Parameters
    ----------
    X_0 : torch.Tensor
        Initial sample positions, shape (N_particles, 2) or compatible.
    W_0 : torch.Tensor
        Initial weights (not used in plotting currently).
    ess_trace : np.ndarray
        Effective Sample Size history, shape (T,).
    rewards_trace : np.ndarray
        Rewards history per particle, shape (T, N_particles).
    particles_trace : list
        List of particle arrays at each time step, each of shape (N_particles, 2).
    log_weights_trace : list
        List of log-weights per time step (not used in current plots).
    num_timesteps : int
        Total number of time steps T.
    vocab_size : int
        Determines axis limits for scatter plots.
    num_categories : int
        Number of output categories for one-hot encoding in reward computation.
    compute_rewards_fn : Callable
        Function that computes rewards given a one-hot tensor of shape (N, num_categories).
    interval : int, optional
        Time step interval between scatter plot snapshots, by default 100.

    Returns
    -------
    None
    """
    # Compute sample diversity (# unique particles) at each time step
    diversity_trace = [
        np.unique(particles_trace[i], axis=0).shape[0]
        for i in range(len(particles_trace))
    ]
    timesteps = np.arange(num_timesteps, -1, -1)

    # Ensure rewards_trace is an array
    rewards_array = np.asarray(rewards_trace)

    # 1) Particle scatter snapshots
    n_plots = num_timesteps // interval + 1
    fig, axes = plt.subplots(nrows=1, ncols=n_plots, figsize=(20, 1.5))
    for i, ax in enumerate(np.atleast_1d(axes)):
        t_idx = i * interval
        samples = particles_trace[t_idx]
        ax.scatter(samples[:, 0], samples[:, 1], s=0.5, alpha=0.7)
        ax.set_title(f"t={num_timesteps - t_idx}")
        ax.set_xlim(-1, vocab_size + 1)
        ax.set_ylim(-1, vocab_size + 1)
        ax.axis('off')
    plt.show()

    # 2) Diversity vs. reward variance
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(timesteps, diversity_trace, linestyle='--', label='Sample Diversity')
    ax1.set_ylabel('Diversity')
    ax1.set_xlabel('Time Step')

    ax2 = ax1.twinx()
    ax2.plot(timesteps, rewards_array.var(axis=1), label='Reward Variance', color='tab:orange')
    ax2.set_ylabel('Variance')
    
    ax1.scatter(resampling_trace, np.zeros(len(resampling_trace)), marker='x', color='red', label='Resamples')
    ax1.invert_xaxis()
    ax1.set_ylim(ymin=0)

    fig.legend(loc='upper right')
    plt.title('Diversity and Reward Variance Over Time')
    plt.show()

    # 3) ESS, Diversity, and Mean Rewards over time
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(timesteps, ess_trace, label='ESS')
    ax1.plot(timesteps, diversity_trace, linestyle='--', label='Sample Diversity', color='tab:green')
    ax1.set_ylabel('ESS / Diversity')
    ax1.set_xlabel('Time Step')

    ax2 = ax1.twinx()
    ax2.plot(timesteps, rewards_array.mean(axis=1), label='Mean Reward', color='tab:red')
    ax2.set_ylabel('Mean Reward')

    ax1.scatter(resampling_trace, np.zeros(len(resampling_trace)), marker='x', color='red', label='Resamples')
    ax1.invert_xaxis()
    ax1.set_ylim(ymin=0)

    fig.legend(loc='upper right')
    plt.title('ESS, Diversity, and Mean Reward Over Time')
    plt.show()
    
    
    # 3) Final sample scatter and metrics
    samples_final = X_0.cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.scatter(samples_final[:, 0], samples_final[:, 1], s=10, alpha=0.5)
    plt.xlim(-1, vocab_size + 1)
    plt.ylim(-1, vocab_size + 1)
    plt.title('Final Samples (X_0)')
    plt.show()

    # Compute final diversity and average reward
    one_hot = F.one_hot(torch.tensor(samples_final, dtype=torch.long), num_classes=num_categories).float()
    avg_reward = compute_rewards_fn(one_hot).mean().item()
    final_diversity = np.unique(samples_final, axis=0).shape[0]
    print(f"Final average reward: {avg_reward:.4f}")
    print(f"Final diversity: {final_diversity}")


def show_binarized_images_with_rewards(imgs, rewards, log_weights, title=None):
    """
    Display a batch of binarized images (0/1) with masked pixels (2) in a single row,
    annotating each with its reward and normalized weight.

    Args:
        imgs (torch.Tensor or np.ndarray): Array of shape (N, 1, 28, 28) with values in {0,1,2}.
            0 = background, 1 = foreground, 2 = masked.
        rewards (torch.Tensor or np.ndarray): Array of shape (N,) with reward values.
        log_weights (torch.Tensor or np.ndarray): Array of shape (N,) with log-weights.
        title (str, optional): Overall title for the figure.
    """
    # to numpy
    if hasattr(imgs, 'cpu'):
        imgs = imgs.cpu().numpy()
    if hasattr(rewards, 'cpu'):
        rewards = rewards.cpu().numpy()
    if hasattr(log_weights, 'cpu'):
        log_weights = log_weights.cpu().numpy()

    N = imgs.shape[0]
    fig, axes = plt.subplots(1, N, figsize=(N, 2))
    if title:
        fig.suptitle(title, fontsize=12)

    # Define discrete colormap: 0→black, 1→white, 2→blue
    cmap = mcolors.ListedColormap(['black', 'white', 'blue'])
    norm = mcolors.BoundaryNorm([-.5, .5, 1.5, 2.5], ncolors=3)

    for i, ax in enumerate(axes):
        img = imgs[i].squeeze(0)  # now shape (28,28)
        ax.imshow(img, cmap=cmap, norm=norm, interpolation='nearest')
        weight = np.exp(log_weights[i])
        ax.set_title(f"r={rewards[i]:.3f}\nw={weight:.2f}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def binarized_images_diversity(imgs):
    """
    imgs: LongTensor of shape (B,1,28,28) with values {0,1,2}.
          2 = masked (ignore).
    returns: scalar—mean normalized Hamming distance over all image pairs.
    """
    imgs = torch.from_numpy(imgs)
    B = imgs.size(0)
    # flatten to (B, 28*28)
    x = imgs.view(B, -1)
    # mask where !=2
    valid = (x != 2)  # (B, P)
    
    # preallocate
    total = 0.0
    count = 0

    for i in range(B):
        xi, mi = x[i], valid[i]
        for j in range(i+1, B):
            xj, mj = x[j], valid[j]
            # only compare pixels both unmasked
            both = mi & mj                  # (P,)
            n = both.sum().item()
            if n == 0:
                continue  # no common pixels—skip
            # XOR counts where bits differ
            diffs = (xi[both] != xj[both]).sum().item()
            total += diffs / n
            count += 1

    val = total / count if count > 0 else 0.0
    return val


def plot_smc_results_binarized_mnist(
    X_0: torch.Tensor,
    W_0: torch.Tensor,
    ess_trace: np.ndarray,
    rewards_trace: np.ndarray,
    particles_trace: list,
    log_weights_trace: list,
    resampling_trace: list,
    num_timesteps: int,
    vocab_size: int,
    num_categories: int,
    compute_rewards_fn: Callable[[torch.Tensor], torch.Tensor],
    interval: int = 100,
    diversity_score_fn: Callable = binarized_images_diversity,
) -> None:
    """
    Visualize the progression and diagnostics of a Sequential Monte Carlo (SMC) run.

    This function generates several plots:
    1. Scatter plots of particles at different time steps.
    2. Combined plot showing sample diversity and reward variance over time.
    3. Final scatter of initial samples X_0 with computed diversity and average reward.
    4. Combined plot of effective sample size (ESS), sample diversity, and mean rewards over time.

    Parameters
    ----------
    X_0 : torch.Tensor
        Initial sample positions, shape (N_particles, 2) or compatible.
    W_0 : torch.Tensor
        Initial weights (not used in plotting currently).
    ess_trace : np.ndarray
        Effective Sample Size history, shape (T,).
    rewards_trace : np.ndarray
        Rewards history per particle, shape (T, N_particles).
    particles_trace : list
        List of particle arrays at each time step, each of shape (N_particles, 2).
    log_weights_trace : list
        List of log-weights per time step (not used in current plots).
    num_timesteps : int
        Total number of time steps T.
    vocab_size : int
        Determines axis limits for scatter plots.
    num_categories : int
        Number of output categories for one-hot encoding in reward computation.
    compute_rewards_fn : Callable
        Function that computes rewards given a one-hot tensor of shape (N, num_categories).
    interval : int, optional
        Time step interval between scatter plot snapshots, by default 100.

    Returns
    -------
    None
    """
    # Compute sample diversity (# unique particles) at each time step
    diversity_trace = [
        diversity_score_fn(particles_trace[i])
        for i in range(len(particles_trace))
    ]
    timesteps = np.arange(num_timesteps, -1, -1)

    # Ensure rewards_trace is an array
    rewards_array = np.asarray(rewards_trace)
    
    
    # 1) ESS, Diversity, and Mean Rewards over time
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(timesteps, ess_trace, label='ESS')
    # ax1.plot(timesteps, diversity_trace, linestyle='--', label='Sample Diversity', color='tab:green')
    ax1.set_ylabel('ESS')
    ax1.set_xlabel('Time Step')

    ax2 = ax1.twinx()
    ax2.plot(timesteps, rewards_array.mean(axis=1), label='Mean Reward', color='tab:red')
    ax2.set_ylabel('Mean Reward')

    ax1.scatter(resampling_trace, np.zeros(len(resampling_trace)), marker='x', color='red', label='Resamples')
    ax1.invert_xaxis()
    ax1.set_ylim(ymin=0)

    fig.legend(loc='upper right')
    plt.title('ESS, Diversity, and Mean Reward Over Time')
    plt.show()
    
    
    # 2) Diversity vs. reward variance
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(timesteps, diversity_trace, linestyle='--', label='Diversity score')
    ax1.set_ylabel('Diversity score')
    ax1.set_xlabel('Time Step')

    ax2 = ax1.twinx()
    ax2.plot(timesteps, rewards_array.var(axis=1), label='Reward Variance', color='tab:orange')
    ax2.set_ylabel('Variance')
    
    ax1.scatter(resampling_trace, np.zeros(len(resampling_trace)), marker='x', color='red', label='Resamples')
    ax1.invert_xaxis()
    ax1.set_ylim(ymin=0)

    fig.legend(loc='upper right')
    plt.title('Diversity and Reward Variance Over Time')
    plt.show()
    

    # 3) Particle scatter snapshots
    step = num_timesteps // 10
    timesteps_to_plot = [i * step for i in range(11)]  # [0, step, 2*step, …, 10*step]
    for t in timesteps_to_plot:
        imgs = particles_trace[t]             # shape: (100,1,28,28)
        rewards = rewards_trace[t]           # shape: (100,)
        log_weights = log_weights_trace[t]   # shape: (100,)
        show_binarized_images_with_rewards(imgs, rewards, log_weights, title=f'Timestep {num_timesteps - t}')
    
    # 4) Final sample scatter and metrics
    samples_final = X_0.cpu().numpy()
    one_hot = F.one_hot(torch.tensor(samples_final, dtype=torch.long), num_classes=num_categories).float()
    rewards = compute_rewards_fn(one_hot).cpu()
    log_weights = np.log(W_0.cpu().numpy())
    show_binarized_images_with_rewards(samples_final, rewards, log_weights, title="Final resampled particles")

    # Compute final diversity and average reward
    avg_reward = rewards.mean().item()
    final_uniqueness = np.unique(samples_final, axis=0).shape[0]
    final_diversity = diversity_score_fn(samples_final)
    print(f"Final average reward: {avg_reward:.4f}")
    print(f"Final diversity score: {final_diversity}")
    print(f"Final uniqueness: {final_uniqueness}")
