from tqdm import tqdm
import torch
import torch.nn.functional as F
from .utils import (
    compute_ess_from_log_w,
    normalize_log_weights, 
    normalize_weights
)


def sequential_monte_carlo(
    model, 
    num_categories,
    T,
    N, 
    ESS_min, 
    intialize_particles_fn,
    resample_fn, 
    proposal_fn,
    compute_reward_fn,
    lambdas,
    kl_weight,
    reward_estimate_sample_count=None,
    eps=1e-9,
    device='cpu',
    verbose=False,
):
    """
    Perform Sequential Monte Carlo (SMC) with variable resampling.

    Args:
        model (nn.Module): the pretrained diffusion model
        mask_token (int): the integer representing mask category
        T (int): Number of time steps.
        N (int): Number of particles.
        ESS_min (int): Minimum effective sample size.
        resample_fn (function): Resampling function.
        proposal_fn (function): Generates new particles using some proposal
        lambdas (list): List of lambda values.
        reward_estimate (str): argmax/sampling/mean
        reward_estimate_sample_count (int): only applicable if reward_estimate = sampling

    Returns:
        list: particle approximations [(x^i, w^i)]
    """
    model.eval()

    # Initialize particles at time T
    X_t = intialize_particles_fn(N, device=device)
    log_W_t = torch.zeros(N, device=device, requires_grad=False)
    
    log_prob_diffusion = torch.zeros(N).to(device)
    log_prob_proposal = torch.zeros(N).to(device)
    log_twist_func_prev = torch.zeros(N).to(device)
    
    particles_trace = [X_t.cpu().numpy()]
    log_weights_trace = []
    ess_trace = []
    rewards_trace = []
    resample_count = 0
    
    for t in tqdm(range(T, 0, -1)):
        # Compute rewards and rewards grad
        z_t = F.one_hot(X_t, num_classes=num_categories).float()
        z_t.requires_grad_()
        x_s_probs, x0_probs = model.sample_step(z_t, t, device=device)
        x0_samples = F.gumbel_softmax(
            logits=torch.log(x0_probs + eps).unsqueeze(1).expand(-1, reward_estimate_sample_count, -1, -1),
            hard=True,
        )
        rewards = compute_reward_fn(x0_samples.reshape(N * reward_estimate_sample_count, 2, num_categories)).reshape(N, reward_estimate_sample_count)
        rewards = rewards.mean(dim=1)
        rewards_grad = torch.autograd.grad(outputs=rewards, inputs=z_t, grad_outputs=torch.ones_like(rewards))[0]
        
        # After computing gradients, detach and other tensors if needed
        x_s_probs = x_s_probs.detach()
        rewards = rewards.detach()
        
        rewards_trace.append(rewards.cpu())
        
        log_twist_func = (lambdas[t] / kl_weight) * rewards
        
        # Update weights
        log_W_t += (
            log_prob_diffusion - log_prob_proposal +
            log_twist_func - log_twist_func_prev
        )
        log_W_t = normalize_log_weights(log_W_t)
        if torch.isnan(log_W_t).any() or torch.isinf(log_W_t).any():
            print("NaN or Inf encountered in log_W_t")
            raise ValueError("NaN or Inf encountered in log_W_t")
        log_weights_trace.append(log_W_t.cpu().numpy())
        
        # Adaptive resampling based on ESS
        ESS = compute_ess_from_log_w(log_W_t)
        ess_trace.append(ESS.item())
        if ESS < ESS_min:
            resampled_indices = resample_fn(log_W_t)
            X_t = X_t[resampled_indices]
            log_W_t = log_W_t.zero_()
            x_s_probs = x_s_probs[resampled_indices]
            log_twist_func = log_twist_func[resampled_indices]
            rewards_grad = rewards_grad[resampled_indices]
            if verbose:
                print(f"Resampled at step {t}")
            resample_count += 1

        
        # Update particles using proposal
        X_t, log_prob_proposal = proposal_fn(X_t, x_s_probs, t, lambdas, kl_weight, rewards_grad, model, mask_token, reward_estimate_sample_count)
        particles_trace.append(X_t.cpu().numpy())
        
        diffusion_distribution = torch.distributions.Categorical(probs=x_s_probs)
        log_prob_diffusion = diffusion_distribution.log_prob(X_t).sum(dim=1)
        
        # assert torch.allclose(log_prob_diffusion, log_prob_proposal), "Log probabilities do not match"
        
        log_twist_func_prev = log_twist_func
    
    final_rewards = compute_reward_fn(F.one_hot(X_t, num_classes=num_categories).float())
    rewards_trace.append(final_rewards.cpu())
    log_twist_func = (lambdas[0] / kl_weight) * final_rewards
    # Update weights
    log_W_t += (
        log_prob_diffusion - log_prob_proposal +
        log_twist_func - log_twist_func_prev
    )
    log_W_t = normalize_log_weights(log_W_t)
    log_weights_trace.append(log_W_t.cpu().numpy())

    # Final resampling to get uniform weights
    resampled_indices = resample_fn(log_W_t)
    X_t = X_t[resampled_indices]
    log_W_t = torch.zeros_like(log_W_t)
    
    print(f"Resampled {resample_count} times.")

    return X_t, normalize_weights(log_W_t), ess_trace, rewards_trace, particles_trace, log_weights_trace
