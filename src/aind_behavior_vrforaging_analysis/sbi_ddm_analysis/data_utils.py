"""
Data generation utilities for training and validating MNLE on patch foraging.

This module provides:
1. WindowDataset - training data with drift around mean θ
2. ValidationSession - sessions with known θ(t) for testing recovery
3. Sliding window extraction utilities
4. Validation metrics
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM, create_prior


class WindowDataset(Dataset):
    """
    PyTorch Dataset for training MNLE.
    
    Generates windows where θ drifts around a mean value.
    MNLE learns to infer p(theta_mean | window_data).
    """
    
    def __init__(self,
                 simulator: PatchForagingDDM,
                 num_windows: int,
                 window_sites: int = 100,
                 drift_sigma: float = 0.05,
                 patches_per_window: int = 10,
                 prior_low: torch.Tensor = None,
                 prior_high: torch.Tensor = None,
                 seed: Optional[int] = None):
        """
        Args:
            simulator: PatchForagingDDM instance
            num_windows: number of training windows to generate
            window_sites: sites per window
            drift_sigma: random walk std per patch
            patches_per_window: expected patches per window
            prior_low/high: bounds for sampling theta_mean
            seed: random seed for reproducibility
        """
        self.simulator = simulator
        self.num_windows = num_windows
        self.window_sites = window_sites
        self.drift_sigma = drift_sigma
        self.patches_per_window = patches_per_window
        
        # Prior bounds
        if prior_low is None:
            prior_low = torch.tensor([0.01, 0.01, 0.0])
        if prior_high is None:
            prior_high = torch.tensor([2.0, 2.0, 2.0])
        self.prior_low = prior_low
        self.prior_high = prior_high
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Pre-generate dataset
        print(f"Generating {num_windows} training windows...")
        self.windows = []
        self.theta_means = []
        
        for i in range(num_windows):
            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1}/{num_windows}")
            
            # Sample theta_mean from prior
            theta_mean = self.prior_low + torch.rand(3) * (self.prior_high - self.prior_low)
            
            # Generate window with drift (guaranteed to be window_sites length)
            window_data, _, _ = simulator.simulate_window_with_drift(
                theta_mean=theta_mean,
                window_sites=window_sites,
                drift_sigma=drift_sigma,
                patches_per_window=patches_per_window
            )
            
            assert len(window_data) == window_sites, f"Window size mismatch: {len(window_data)} != {window_sites}"
            
            self.windows.append(window_data)
            self.theta_means.append(theta_mean)
        
        self.windows = torch.stack(self.windows)
        self.theta_means = torch.stack(self.theta_means)
        
        print(f"Dataset complete: {self.windows.shape}, {self.theta_means.shape}")
    
    def __len__(self):
        return self.num_windows
    
    def __getitem__(self, idx):
        return self.windows[idx], self.theta_means[idx]
    
    def get_statistics(self):
        """Return dataset statistics for diagnostics."""
        stats = {
            'num_windows': len(self),
            'window_sites': self.window_sites,
            'theta_mean_ranges': {
                'drift_rate': (self.theta_means[:, 0].min().item(), 
                              self.theta_means[:, 0].max().item()),
                'reward_bump': (self.theta_means[:, 1].min().item(), 
                               self.theta_means[:, 1].max().item()),
                'failure_bump': (self.theta_means[:, 2].min().item(), 
                                self.theta_means[:, 2].max().item()),
            },
            'avg_rewards_per_window': self.windows[:, :, 1].sum(dim=1).mean().item(),
            'avg_stops_per_window': self.windows[:, :, 2].sum(dim=1).mean().item(),
        }
        return stats


class ValidationSession:
    """
    Validation session with known time-varying θ(t).
    
    Used to test if sliding window inference can recover ground truth trajectories.
    """
    
    def __init__(self,
                 simulator: PatchForagingDDM,
                 trajectory_type: str = 'random_walk',
                 n_patches: int = 50,
                 theta_init: torch.Tensor = None,
                 **trajectory_kwargs):
        """
        Args:
            simulator: PatchForagingDDM instance
            trajectory_type: 'random_walk' or 'step_change'
            n_patches: number of patches in session
            theta_init: initial parameters (random if None)
            **trajectory_kwargs: passed to trajectory generator
                For random_walk: sigma (default 0.1)
                For step_change: n_changes (default 3)
        """
        self.simulator = simulator
        self.trajectory_type = trajectory_type
        self.n_patches = n_patches
        
        # Generate initial theta if not provided
        if theta_init is None:
            theta_init = torch.tensor([0.01, 0.01, 0.0]) + \
                        torch.rand(3) * (torch.tensor([2.0, 2.0, 2.0]) - torch.tensor([0.01, 0.01, 0.0]))
        
        # Generate trajectory
        if trajectory_type == 'random_walk':
            sigma = trajectory_kwargs.get('sigma', 0.1)
            self.trajectory = simulator.generate_random_walk_trajectory(
                n_patches=n_patches,
                theta_init=theta_init,
                sigma=sigma
            )
        elif trajectory_type == 'step_change':
            n_changes = trajectory_kwargs.get('n_changes', 3)
            self.trajectory = simulator.generate_step_change_trajectory(
                n_patches=n_patches,
                n_changes=n_changes
            )
        else:
            raise ValueError(f"Unknown trajectory_type: {trajectory_type}")
        
        # Simulate session
        self.session_data, self.theta_per_site, self.patch_boundaries = \
            simulator.simulate_session_with_trajectory(self.trajectory)
        
        print(f"ValidationSession created:")
        print(f"  Type: {trajectory_type}")
        print(f"  Patches: {n_patches}")
        print(f"  Total sites: {len(self.session_data)}")
        print(f"  Sites per patch: {len(self.session_data) / n_patches:.1f}")
    
    def get_ground_truth_for_site(self, site_idx: int) -> torch.Tensor:
        """Get the true θ that generated a specific site."""
        return self.theta_per_site[site_idx]
    
    def get_ground_truth_for_patch(self, patch_idx: int) -> torch.Tensor:
        """Get the true θ for a specific patch."""
        return self.trajectory[patch_idx]
    
    def plot_trajectory(self, save_path: Optional[str] = None):
        """Plot the ground truth trajectory."""
        import matplotlib.pyplot as plt
        
        trajectory_array = torch.stack(self.trajectory).numpy()
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        param_names = ['drift_rate', 'reward_bump', 'failure_bump']
        
        for i, (ax, name) in enumerate(zip(axes, param_names)):
            ax.plot(trajectory_array[:, i], linewidth=2, marker='o', markersize=4)
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{name} - {self.trajectory_type}')
        
        axes[-1].set_xlabel('Patch Index')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Trajectory plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def extract_sliding_windows(session_data: torch.Tensor,
                            theta_per_site: torch.Tensor,
                            window_sites: int = 100,
                            stride: int = 50) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Extract overlapping windows from a session with known θ(t).
    
    Args:
        session_data: (n_sites, 3) tensor from session
        theta_per_site: (n_sites, 3) tensor of true θ at each site
        window_sites: size of each window
        stride: step size between windows (smaller = more overlap)
    
    Returns:
        windows: list of (window_sites, 3) tensors
        ground_truth_means: list of mean θ for each window
    """
    n_sites = len(session_data)
    windows = []
    ground_truth_means = []
    
    start_idx = 0
    while start_idx + window_sites <= n_sites:
        end_idx = start_idx + window_sites
        
        # Extract window
        window = session_data[start_idx:end_idx]
        windows.append(window)
        
        # Compute ground truth: mean θ during this window
        theta_window = theta_per_site[start_idx:end_idx]
        theta_mean = theta_window.mean(dim=0)
        ground_truth_means.append(theta_mean)
        
        start_idx += stride
    
    print(f"Extracted {len(windows)} windows (window_sites={window_sites}, stride={stride})")
    print(f"  Coverage: {start_idx}/{n_sites} sites ({100*start_idx/n_sites:.1f}%)")
    
    return windows, ground_truth_means


def compute_coverage(estimated_samples: torch.Tensor,
                    true_value: torch.Tensor,
                    credible_level: float = 0.95) -> np.ndarray:
    """
    Compute if true_value falls within credible interval of samples.
    
    Args:
        estimated_samples: (n_samples, 3) posterior samples
        true_value: (3,) ground truth
        credible_level: credible interval level
    
    Returns:
        coverage: (3,) boolean array indicating coverage for each parameter
    """
    alpha = (1 - credible_level) / 2
    lower = torch.quantile(estimated_samples, alpha, dim=0)
    upper = torch.quantile(estimated_samples, 1 - alpha, dim=0)
    
    coverage = (true_value >= lower) & (true_value <= upper)
    return coverage.numpy()


def compute_mse(estimated_mean: torch.Tensor,
                true_value: torch.Tensor) -> float:
    """
    Compute mean squared error between estimate and truth.
    
    Args:
        estimated_mean: (3,) posterior mean
        true_value: (3,) ground truth
    
    Returns:
        mse: scalar MSE
    """
    return ((estimated_mean - true_value) ** 2).mean().item()


def compute_validation_metrics(posterior_samples_list: List[torch.Tensor],
                               ground_truth_list: List[torch.Tensor],
                               credible_level: float = 0.95) -> dict:
    """
    Compute validation metrics across multiple windows.
    
    Args:
        posterior_samples_list: list of (n_samples, 3) tensors, one per window
        ground_truth_list: list of (3,) tensors, one per window
        credible_level: credible interval level
    
    Returns:
        metrics: dict with coverage, MSE, etc.
    """
    n_windows = len(posterior_samples_list)
    coverages = []
    mses = []
    biases = []
    
    for samples, truth in zip(posterior_samples_list, ground_truth_list):
        # Coverage
        cov = compute_coverage(samples, truth, credible_level)
        coverages.append(cov)
        
        # MSE
        mean_estimate = samples.mean(dim=0)
        mse = compute_mse(mean_estimate, truth)
        mses.append(mse)
        
        # Bias
        bias = (mean_estimate - truth).numpy()
        biases.append(bias)
    
    coverages = np.array(coverages)
    biases = np.array(biases)
    
    metrics = {
        'n_windows': n_windows,
        'coverage_per_param': coverages.mean(axis=0),  # (3,) coverage for each param
        'overall_coverage': coverages.all(axis=1).mean(),  # fraction where all 3 params covered
        'mean_mse': np.mean(mses),
        'std_mse': np.std(mses),
        'mean_bias_per_param': biases.mean(axis=0),  # (3,) mean bias per param
        'std_bias_per_param': biases.std(axis=0),  # (3,) std of bias per param
    }
    
    return metrics


def print_validation_metrics(metrics: dict):
    """Pretty print validation metrics."""
    print("\n" + "="*60)
    print("VALIDATION METRICS")
    print("="*60)
    print(f"Number of windows: {metrics['n_windows']}")
    print(f"\nOverall coverage (all params): {metrics['overall_coverage']:.2%}")
    print(f"\nPer-parameter coverage:")
    param_names = ['drift_rate', 'reward_bump', 'failure_bump']
    for name, cov in zip(param_names, metrics['coverage_per_param']):
        print(f"  {name:15s}: {cov:.2%}")
    print(f"\nMean Squared Error:")
    print(f"  Mean: {metrics['mean_mse']:.6f}")
    print(f"  Std:  {metrics['std_mse']:.6f}")
    print(f"\nBias per parameter:")
    for name, bias, std in zip(param_names, 
                               metrics['mean_bias_per_param'], 
                               metrics['std_bias_per_param']):
        print(f"  {name:15s}: {bias:+.4f} ± {std:.4f}")
    print("="*60 + "\n")


# Test functions
if __name__ == "__main__":
    print("Testing data generation utilities...\n")
    
    # Initialize simulator
    simulator = PatchForagingDDM()
    
    # Test 1: WindowDataset
    print("="*60)
    print("Test 1: WindowDataset")
    print("="*60)
    dataset = WindowDataset(
        simulator=simulator,
        num_windows=100,
        window_sites=100,
        drift_sigma=0.05,
        seed=42
    )
    
    # Check a sample
    window, theta_mean = dataset[0]
    print(f"\nSample window shape: {window.shape}")
    print(f"Sample theta_mean: {theta_mean}")
    
    # Statistics
    stats = dataset.get_statistics()
    print(f"\nDataset statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val}")
    
    # Test 2: ValidationSession (random walk)
    print("\n" + "="*60)
    print("Test 2: ValidationSession (random walk)")
    print("="*60)
    session_rw = ValidationSession(
        simulator=simulator,
        trajectory_type='random_walk',
        n_patches=30,
        sigma=0.1
    )
    
    print(f"\nGround truth at site 0: {session_rw.get_ground_truth_for_site(0)}")
    print(f"Ground truth at patch 0: {session_rw.get_ground_truth_for_patch(0)}")
    
    # Test 3: ValidationSession (step change)
    print("\n" + "="*60)
    print("Test 3: ValidationSession (step change)")
    print("="*60)
    session_step = ValidationSession(
        simulator=simulator,
        trajectory_type='step_change',
        n_patches=30,
        n_changes=3
    )
    
    # Test 4: Extract sliding windows
    print("\n" + "="*60)
    print("Test 4: Sliding window extraction")
    print("="*60)
    windows, gt_means = extract_sliding_windows(
        session_data=session_rw.session_data,
        theta_per_site=session_rw.theta_per_site,
        window_sites=100,
        stride=50
    )
    
    print(f"\nFirst window shape: {windows[0].shape}")
    print(f"First ground truth mean: {gt_means[0]}")
    print(f"Last ground truth mean: {gt_means[-1]}")
    
    # Test 5: Mock validation metrics
    print("\n" + "="*60)
    print("Test 5: Validation metrics (with mock posteriors)")
    print("="*60)
    
    # Create mock posterior samples (just for testing the metrics function)
    mock_posteriors = []
    for gt in gt_means[:5]:  # Just test first 5 windows
        # Create fake posterior centered around truth with some noise
        samples = gt.unsqueeze(0) + torch.randn(1000, 3) * 0.1
        mock_posteriors.append(samples)
    
    metrics = compute_validation_metrics(mock_posteriors, gt_means[:5])
    print_validation_metrics(metrics)
    
    print("All tests passed!")