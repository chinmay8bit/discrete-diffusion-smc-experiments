import argparse
import json
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from rich import print

import sys
import os
sys.path.append('.')

from smc.smc import sequential_monte_carlo
from smc.sampling_algorithms import (
    systematic_resample,
    stratified_resample,
    multinomial_resample
)
from smc.utils import lambda_schedule

from smc_scripts.plot_utils import plot_smc_results_binarized_mnist
from smc.proposals import (
    reverse_as_proposal, 
    first_order_approximation_optimal_proposal, 
    first_order_approximation_optimal_proposal_with_gradient_clipping
)
from utils.metadata import get_metadata, save_metadata_json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import random
import torch.backends.cudnn as cudnn

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(42)


from models.discrete_diffusion.mdm import MaskedDiffusion
from models.discrete_diffusion.utils.parametrizations import (
    subs_parametrization,
    subs_parametrization_continuous,
)
from models.denoising_models.mlp import MLP
from models.denoising_models.unet_with_attention import UNetWithAttention

def main(args):
    batch_size = args.batch_size
    vocab_size = 2
    num_categories = vocab_size + 1  #includes the mask category
    mask_index = num_categories - 1
    input_shape = (1, 28, 28)
    num_timesteps = args.num_timesteps


    # load pretrained model
    pretrained_model = MaskedDiffusion(
        denoising_model=UNetWithAttention(
            num_categories=num_categories,
            embedding_dim=64,
            ch_mult=(2, 4, 8),
            num_res_blocks=2,
            attention_resolutions=(1, 2),
            encode_time=False,
            probs_parametrization_fn=subs_parametrization_continuous,
        ),
        num_categories=num_categories,
        input_shape=input_shape,
        mask_index=mask_index,
        masking_schedule=args.masking_schedule,
        num_timesteps=num_timesteps,
        discretization_schedule=args.discretization_schedule,
    ).to(device)

    # load model weights
    pretrained_model.load_state_dict(torch.load('model_weights/mdm_binarized_mnist_256_copy.pth'))
    pretrained_model.eval()
    
    # define target digit
    target_digit = args.target_digit
    
    # define kl weight
    kl_weight = args.kl_weight
    
    # define reward functions
    from models.reward_models.binarized_mnist_classifier import BinarizedMNISTClassifier
    mnist_classfier_model = BinarizedMNISTClassifier().to(device)
    mnist_classfier_model.load_state_dict(torch.load('model_weights/binarized_mnist_classifier_1_copy.pth'))

    def compute_rewards_for_batch(x: Tensor, with_grad=False):
        # x.shape : (B, 1, 28, 28, num_categories)
        logits = mnist_classfier_model(x[..., :vocab_size].to(device)) # Shape: (B, 10)
        logits = logits.log_softmax(dim=-1) # Shape: (B, 10)
        reward = logits[:, target_digit]
        if args.reward_clamp_max is not None:
            reward = reward.clamp_max(args.reward_clamp_max)
        if args.reward_clamp_min is not None:
            reward = reward.clamp_min(args.reward_clamp_min)
        return reward

    def compute_rewards(x, with_grad=False):
        n_samples = x.shape[0]
        rewards_all = []
        for i in range(0, n_samples, batch_size):
            if with_grad:
                rewards = compute_rewards_for_batch(x[i:i + batch_size])
            else:
                with torch.no_grad():
                    rewards = compute_rewards_for_batch(x[i:i + batch_size])
            rewards_all.append(rewards)
        rewards_all = torch.cat(rewards_all)
        return rewards_all
    
    # Define external reward
    from models.reward_models.binarized_mnist_classifier import BinarizedMNISTClassifierExt
    mnist_classfier_model_ext = BinarizedMNISTClassifierExt().to(device)
    mnist_classfier_model_ext.load_state_dict(torch.load('model_weights/binarized_mnist_classifier_ext.pth'))

    def compute_rewards_for_batch_ext(x: Tensor):
        # x.shape : (B, 1, 28, 28, num_categories)
        if x.ndim == 5:
            x = x.argmax(dim=-1).float() # x.shape (B, 1, 28, 28)
        logits = mnist_classfier_model_ext(x.to(device)) # Shape: (B, 10)
        logits = logits.log_softmax(dim=-1) # Shape: (B, 10)
        reward = logits[:, target_digit]
        return reward

    @torch.no_grad()
    def compute_rewards_ext(x):
        n_samples = x.shape[0]
        rewards_all = []
        for i in range(0, n_samples, batch_size):
            rewards = compute_rewards_for_batch_ext(x[i:i + batch_size])
            rewards_all.append(rewards)
        rewards_all = torch.cat(rewards_all)
        return rewards_all

    def get_ext_embeddings_batch(x):
        # x.shape : (B, 1, 28, 28, num_categories)
        if x.ndim == 5:
            x = x.argmax(dim=-1).float() # x.shape (B, 1, 28, 28)
        embeds = mnist_classfier_model_ext.get_embeddings(x.to(device)) # Shape: (B, 64)
        return embeds

    @torch.no_grad()
    def get_ext_embeddings(x):
        n_samples = x.shape[0]
        embeds_all = []
        for i in range(0, n_samples, batch_size):
            embeds = get_ext_embeddings_batch(x[i:i + batch_size])
            embeds_all.append(embeds)
        embeds_all = torch.cat(embeds_all)
        return embeds_all
    
    # define initalize particles fn
    def intialize_particles(num_particles, device=device):
        particles = torch.full((num_particles, *input_shape), mask_index, device=device, requires_grad=False)
        return particles
    
    # Define the lambdas schedule and other parameters
    num_particles = args.num_particles
    ESS_min = args.ESS_min if args.ESS_min is not None else num_particles // 2

    lambda_schedule_type = args.lambda_schedule_type
    if lambda_schedule_type == "exp":
        lambda_schedule_exp = args.lambda_schedule_exp
        lambdas = lambda_schedule(num_timesteps, gamma=lambda_schedule_exp - 1)
    elif lambda_schedule_type == "linear":
        lambda_one_after = args.lambda_one_after
        lambdas = torch.cat([torch.ones(num_timesteps - lambda_one_after), torch.linspace(1, 0, lambda_one_after + 1)])
    else:
        raise ValueError("Invalid lambda_schedule_type")
    
    reward_estimate_sample_count = args.phi
    use_partial_resampling = args.use_partial_resampling
    perform_final_resample = args.perform_final_resample
    partial_resample_size = args.partial_resample_size if args.partial_resample_size is not None else num_particles // 2
    
    runs_per_method = args.runs_per_method
    
    # Collect all metadata and save to json
    metadata = get_metadata(dict(locals()), ignore_internal=True)
    print(metadata)
    # Save args as well
    args_dict = vars(args)
    
    base_dir = "smc_scripts/outputs/smc_mdm_binarized_mnist_multi_runs"
    
    # Check if exact same args.json exists
    def find_existing_args():
        """Search recursively under base_dir for an args.json that matches args_dict.
        Return the directory containing the matching args.json, or None if not found."""
        if not os.path.isdir(base_dir):
            return None

        for root, _, files in os.walk(base_dir):
            if "args.json" in files:
                path = os.path.join(root, "args.json")
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        existing_args = json.load(f)
                except Exception as e:
                    # warn and skip corrupted/unreadable files
                    print(f"Warning: could not read {path}: {e}", file=sys.stderr)
                    continue

                # exact dict equality check
                if existing_args == args_dict:
                    return root
        return None

    existing_dir = find_existing_args()
    if existing_dir:
        print(f"Experiment with args already exists in {existing_dir}")
        sys.exit(0)
    
    cur_time = datetime.now().strftime("%Y%m%d/%H%M%S")
    outputs_dir = os.path.join(base_dir, cur_time)
    os.makedirs(outputs_dir, exist_ok=True)

    save_metadata_json(metadata, outputs_dir)
    with open(os.path.join(outputs_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)

    BASE_MODEL = "BASE_MODEL"
    IS_10K = "IS_10K"
    # SMC_LOP = "SMC_LOP"
    SMC_RP = "SMC_RP"
    SMC_FALOP = "SMC_FALOP"

    all_experiment_results = {
        BASE_MODEL: [],
        # SMC_LOP: [],
        SMC_RP: [],
        SMC_FALOP: [],
        IS_10K: [],
    }
    
    
    # Utility methods
    def effective_rank(X: torch.Tensor) -> torch.Tensor:
        """
        Compute the effective rank (Roy & Vetterli, 2007) of a matrix.
        https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf
        
        Args:
            X: (..., n, d) tensor (batch of matrices or a single matrix).
            Typically X is a set of vectors (n,d), and we compute covariance.
            
        Returns:
            erank: effective rank scalar (or batch of scalars).
        """
        # Compute covariance (n,d) -> (d,d)
        if X.ndim == 2:
            C = X.T @ X
        else:
            raise ValueError("Expected 2D tensor (n,d)")
        
        # Eigen decomposition
        eigvals = torch.linalg.eigvalsh(C)
        eigvals = torch.clamp(eigvals, min=0)  # numerical safety
        
        if eigvals.sum() == 0:
            return torch.tensor(0.0, device=X.device)
        
        # Normalize to probability distribution
        p = eigvals / eigvals.sum()
        
        # Shannon entropy
        entropy = -(p * (p+1e-12).log()).sum()
        
        # Effective rank
        return torch.exp(entropy)


    def calculate_metrics(smc_results=None, final_samples=None):
        final_samples = smc_results["X_0"]  if smc_results is not None else final_samples
        assert final_samples is not None
        rewards = compute_rewards(
            F.one_hot(final_samples.long(), num_classes=num_categories).float()
        )
        accuracy = (rewards.exp() > 0.5).float().mean().item()
        rewards_ext = compute_rewards_ext(final_samples.float())
        accuracy_ext = (rewards_ext.exp() > 0.5).float().mean().item()
        
        embeddings_ext = get_ext_embeddings(final_samples.float()) # (B, 64)
        diversity = effective_rank(embeddings_ext).item()
        
        if smc_results is not None:
            resamples = len(smc_results["resampling_trace"])
            if perform_final_resample:
                resamples -= 1 # Final resampling is get particles with uniform weights, its not triggered due to low ESS
        else:
            resamples = 0
        
        return {
            "accuracy": accuracy,
            "accuracy_ext": accuracy_ext,
            "reward": rewards.mean().item(),
            "reward_ext": rewards_ext.mean().item(),
            "diversity": diversity,
            "resamples": resamples,
        }
        
    def mean_and_std(values):
        mean = np.mean(values)
        std = np.std(values)
        return mean, std

    def summarize_metrics(run_list):
        metric_list = ["accuracy", "accuracy_ext", "reward", "reward_ext", "diversity", "resamples"]
        metric_labels = {
            "accuracy": "Accuracy",
            "accuracy_ext": "Accuracy (Ext)",
            "reward": "Reward",
            "reward_ext": "Reward (Ext)",
            "diversity": "Diversity",
            "resamples": "Resample Count"
        }
        metric_dict = {}
        for run in run_list:
            metrics = run["metrics"]
            for metric in metric_list:
                if metric not in metric_dict:
                    metric_dict[metric] = []
                metric_dict[metric].append(metrics[metric])
                
        print(f"Number of runs: {len(run_list)}")
        for metric in metric_list:
            values = metric_dict[metric]
            mean, std = mean_and_std(values)
            print(f"{metric_labels[metric]}: {mean:.2f} ± {std:.2f}")
    
    
    
    
    # Base model runs
    print("Running base model...")
    for run in tqdm(range(runs_per_method)):
        with torch.no_grad():
            samples = pretrained_model.sample(num_samples=num_particles, device=device).float().cpu()
        metrics = calculate_metrics(final_samples=samples)
        all_experiment_results[BASE_MODEL].append(
            {
                "metrics": metrics,
                "results": {
                    "X_0": samples.cpu().numpy(),
                }
            }
        )
    summarize_metrics(all_experiment_results[BASE_MODEL])
    
    
    # Importance sampling with 10K samples
    print("Running IS with 10K samples...")
    
    def target_distribution_log_pdf(x, kl_weight):
        reward = compute_rewards(F.one_hot(x, num_classes=num_categories).float())
        return reward / kl_weight

    def sample_target_distribution(n_samples, kl_weight):
        # 10,000 samples generated using the pretrained model
        dataset = torch.load("datasets_local/pretrained_mnist_samples_dataset.pt", weights_only=False)
        train_loader = DataLoader(dataset, batch_size)
        log_pdf_values = []
        for samples, _ in train_loader:
            samples = samples.long()
            log_pdf_values.append(target_distribution_log_pdf(samples, kl_weight))
        log_pdf_values = torch.cat(log_pdf_values, dim=0)
        indices = torch.distributions.Categorical(logits=log_pdf_values).sample((n_samples,))
        return torch.stack([train_loader.dataset[i][0].long() for i in indices])
    
    for run in tqdm(range(runs_per_method)):
        samples = sample_target_distribution(num_particles, kl_weight)
        metrics = calculate_metrics(final_samples=samples)
        all_experiment_results[IS_10K].append(
            {
                "metrics": metrics,
                "results": {
                    "X_0": samples.cpu().numpy(),
                }
            }
        )
    summarize_metrics(all_experiment_results[IS_10K])
    
    
    # SMC with reverse model proposal
    print("Running SMC with reverse model proposal...")
    for run in tqdm(range(runs_per_method)):
        results = sequential_monte_carlo(
            model=pretrained_model,
            num_categories=num_categories,
            T=num_timesteps,
            N=num_particles,
            ESS_min=ESS_min,
            intialize_particles_fn=intialize_particles,
            resample_fn=systematic_resample,
            use_partial_resampling=use_partial_resampling,
            partial_resample_size=partial_resample_size,
            proposal_fn=reverse_as_proposal,
            compute_reward_fn=compute_rewards,
            lambdas=lambdas,
            kl_weight=kl_weight,
            reward_estimate_sample_count=reward_estimate_sample_count,
            perform_final_resample=perform_final_resample,
            device=device,
            verbose=False,
        )
        metrics = calculate_metrics(results)
        all_experiment_results[SMC_RP].append(
            {
                "metrics": metrics,
                "results": results
            }
        )
    summarize_metrics(all_experiment_results[SMC_RP])
    
    
    # SMC with first order approximation optimal proposal
    print("Running SMC with first order approximation optimal proposal...")
    for run in tqdm(range(runs_per_method)):
        results = sequential_monte_carlo(
            model=pretrained_model,
            num_categories=num_categories,
            T=num_timesteps,
            N=num_particles,
            ESS_min=ESS_min,
            intialize_particles_fn=intialize_particles,
            resample_fn=systematic_resample,
            use_partial_resampling=use_partial_resampling,
            partial_resample_size=partial_resample_size,
            proposal_fn=first_order_approximation_optimal_proposal,
            compute_reward_fn=compute_rewards,
            lambdas=lambdas,
            kl_weight=kl_weight,
            reward_estimate_sample_count=reward_estimate_sample_count,
            perform_final_resample=perform_final_resample,
            device=device,
            verbose=False,
        )
        metrics = calculate_metrics(results)
        all_experiment_results[SMC_FALOP].append(
            {
                "metrics": metrics,
                "results": results
            }
        )
    summarize_metrics(all_experiment_results[SMC_FALOP])
    
    
    import pickle
    # write to disk
    with open(os.path.join(outputs_dir, "all_experiments.pkl"), "wb") as f:
        pickle.dump(all_experiment_results, f)
        
    # how to read from disk
    # ... = pickle.load(open(os.path.join(outputs_dir, "all_experiments.pkl"), "rb"))
    
    
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_particles", type=int, default=16)
parser.add_argument("--num_timesteps", type=int, default=100)
parser.add_argument("--kl_weight", type=float, default=1.0)
parser.add_argument("--target_digit", type=int, default=0)
parser.add_argument("--masking_schedule", type=str, default="linear", choices=["linear", "cosine"])
parser.add_argument("--discretization_schedule", type=str, default="linear", choices=["linear", "cosine"])
parser.add_argument("--lambda_schedule_type", type=str, default="exp", choices=["exp", "linear"])
parser.add_argument("--lambda_schedule_exp", type=float, default=0.05)
parser.add_argument("--lambda_one_after", type=int, default=100)
parser.add_argument("--ESS_min", type=int, default=None)
parser.add_argument("--reward_clamp_max", type=float, default=None)
parser.add_argument("--reward_clamp_min", type=float, default=None)
parser.add_argument("--phi", type=int, default=1)
parser.add_argument("--use_partial_resampling", action="store_true")
parser.add_argument("--partial_resample_size", type=int, default=None)
parser.add_argument("--perform_final_resample", action="store_true")
parser.add_argument("--runs_per_method", type=int, default=10)
args = parser.parse_args()

if __name__ == "__main__":
    main(args)
