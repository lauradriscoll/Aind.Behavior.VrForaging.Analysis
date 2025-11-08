"""
Validation and diagnostics for SBI inference.

Single purpose: Assess quality of parameter recovery.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snpe.snpe_inference import infer_parameters


def run_sbc(simulator, prior, posterior, num_tests: int = 50, 
            num_samples: int = 1000):
    """
    Simulation-Based Calibration test.
    
    Tests if posterior is well-calibrated by checking if true parameters
    fall uniformly in posterior ranks.
    
    Returns:
        ranks: (num_tests, 3) array of parameter ranks
    """
    print(f"\nRunning SBC with {num_tests} tests...")
    
    ranks = []
    successful = 0
    
    error_counts = {'simulation': 0, 'inference': 0, 'nan': 0, 'other': 0}
    
    for _ in tqdm(range(num_tests * 2)):  # Try extra in case some fail
        if successful >= num_tests:
            break
        
        try:
            # Sample true parameters
            theta_true = prior.sample()
            
            # Simulate data
            try:
                window = simulator.simulate_with_random_walk(theta_true, window_sites=100, random_walk_sigma=0.05)
            except Exception as e:
                error_counts['simulation'] += 1
                if error_counts['simulation'] <= 3:  # Print first 3 errors
                    print(f"\n  Simulation error: {type(e).__name__}: {e}")
                continue

            # Infer
            try:
                samples = infer_parameters(posterior, window, num_samples=num_samples)
            except Exception as e:
                error_counts['inference'] += 1
                if error_counts['inference'] <= 3:  # Print first 3 errors
                    print(f"\n  Inference error: {type(e).__name__}: {e}")
                continue
            
            # Check for NaN
            if torch.isnan(samples).any():
                error_counts['nan'] += 1
                continue
            
            # Compute ranks
            rank = (samples < theta_true).sum(dim=0).numpy()
            ranks.append(rank)
            successful += 1
            
        except Exception as e:
            error_counts['other'] += 1
            if error_counts['other'] <= 3:  # Print first 3 errors
                print(f"\n  Other error: {type(e).__name__}: {e}")
            continue
    
    ranks = np.array(ranks)
    
    print(f"SBC complete: {successful}/{num_tests} tests succeeded")
    
    # Report errors
    if successful < num_tests:
        print(f"\nError summary:")
        print(f"  Simulation errors: {error_counts['simulation']}")
        print(f"  Inference errors:  {error_counts['inference']}")
        print(f"  NaN samples:       {error_counts['nan']}")
        print(f"  Other errors:      {error_counts['other']}")
    
    # Only print statistics if we have successful tests
    if successful > 0:
        print(f"\nRank statistics (should be ~{num_samples/2} if well-calibrated):")
        param_names = ['drift_rate', 'reward_bump', 'failure_bump']
        for i, name in enumerate(param_names):
            print(f"  {name:15s}: mean={ranks[:, i].mean():6.1f}, std={ranks[:, i].std():5.1f}")
    else:
        print("\nâš ï¸  No successful tests - cannot compute rank statistics")
    
    return ranks


def compute_correlation_matrix(samples: torch.Tensor):
    """
    Compute parameter correlation matrix from posterior samples.
    
    Returns:
        corr_matrix: (3, 3) correlation matrix
    """
    samples_np = samples.cpu().numpy()
    corr_matrix = np.corrcoef(samples_np.T)
    return corr_matrix


def print_correlations(samples: torch.Tensor):
    """Print correlation matrix with warnings for high correlations."""
    corr = compute_correlation_matrix(samples)
    param_names = ['drift_rate', 'reward_bump', 'failure_bump']
    
    print("\nPosterior Correlation Matrix:")
    print("="*60)
    print("                ", "  ".join([f"{name:12s}" for name in param_names]))
    for i, name in enumerate(param_names):
        print(f"{name:15s}", "  ".join([f"{corr[i,j]:12.3f}" for j in range(3)]))
    print("="*60)
    
    # Check for high correlations
    high_corr = []
    for i in range(3):
        for j in range(i+1, 3):
            if abs(corr[i, j]) > 0.7:
                high_corr.append((param_names[i], param_names[j], corr[i, j]))
    
    if high_corr:
        print("\n⚠️  WARNING: High correlations detected!")
        print("These parameters may not be well-identified:")
        for name1, name2, corr_val in high_corr:
            print(f"  {name1} ↔ {name2}: r = {corr_val:.3f}")
    else:
        print("\n✓ All correlations < 0.7 (parameters well-separated)")
    
    return corr


def plot_posterior(samples: torch.Tensor, true_theta=None, save_path: str = None):
    """Plot posterior distributions."""
    samples_np = samples.cpu().numpy()
    param_names = ['drift_rate', 'reward_bump', 'failure_bump']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(samples_np[:, i], bins=50, density=True, alpha=0.7, edgecolor='black')
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.set_title(f'Posterior: {name}')
        
        if true_theta is not None:
            ax.axvline(true_theta[i].item(), color='red', linestyle='--', 
                      linewidth=2, label='True value')
            ax.legend()
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Posterior plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_pairplot(samples: torch.Tensor, true_theta=None, save_path: str = None):
    """Plot pairwise posterior distributions showing correlations."""
    samples_np = samples.cpu().numpy()
    param_names = ['drift_rate', 'reward_bump', 'failure_bump']
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: marginal distributions
                ax.hist(samples_np[:, i], bins=30, density=True, alpha=0.7, 
                       color='blue', edgecolor='black')
                if true_theta is not None:
                    ax.axvline(true_theta[i].item(), color='red', 
                              linestyle='--', linewidth=2)
                ax.set_ylabel('Density')
            elif i > j:
                # Lower triangle: scatter plots
                ax.scatter(samples_np[:, j], samples_np[:, i], alpha=0.3, s=1)
                if true_theta is not None:
                    ax.scatter(true_theta[j].item(), true_theta[i].item(),
                             color='red', s=100, marker='*', zorder=10)
                
                # Correlation
                corr = np.corrcoef(samples_np[:, j], samples_np[:, i])[0, 1]
                ax.text(0.05, 0.95, f'r={corr:.2f}', transform=ax.transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                # Upper triangle: hide
                ax.axis('off')
            
            # Labels
            if i == 2:
                ax.set_xlabel(param_names[j])
            if j == 0 and i > 0:
                ax.set_ylabel(param_names[i])
            
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Pairplot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


# Test
if __name__ == "__main__":
    print("Testing validation module...")
    
    from simulator import PatchForagingDDM, create_prior
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snpe.snpe_inference import train_sbi, infer_parameters
    
    # Train small model
    print("\n1. Training small model...")
    simulator = PatchForagingDDM()
    prior = create_prior()
    posterior = train_sbi(simulator, prior, num_simulations=500)
    
    # Test SBC
    print("\n2. Testing SBC...")
    ranks = run_sbc(simulator, prior, posterior, num_tests=10)
    print(f"   Ranks shape: {ranks.shape}")
    
    # Test inference and correlation
    print("\n3. Testing correlation analysis...")
    test_window = simulator.simulate_constant(torch.tensor([0.5, 0.6, 0.2]), 100)
    samples = infer_parameters(posterior, test_window, num_samples=500)
    corr = print_correlations(samples)
    
    # Test plotting
    print("\n4. Testing plots...")
    plot_posterior(samples, true_theta=torch.tensor([0.5, 0.6, 0.2]), 
                  save_path='test_posterior.png')
    plot_pairplot(samples, true_theta=torch.tensor([0.5, 0.6, 0.2]),
                 save_path='test_pairplot.png')
    
    print("\n✓ All tests passed!")