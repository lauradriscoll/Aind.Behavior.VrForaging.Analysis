"""
Diagnostic tools for SBI inference.

Implements:
1. Simulation-Based Calibration (SBC) - test if inference is calibrated
2. Pairplot analysis - visualize posterior correlations
3. Prior vs posterior comparison
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from tqdm import tqdm
from contextlib import contextmanager

@contextmanager
def maybe_force_mcmc(posterior, use_mcmc=True):
    """
    Temporarily force MCMC sampling if the posterior supports it.
    Prints which posterior type is in use and safely restores state.
    """
    posterior_type = type(posterior).__name__

    print(f"\n[maybe_force_mcmc] Posterior type detected: {posterior_type}")
    if use_mcmc:
        print("[maybe_force_mcmc] Requested: use MCMC sampling")

    if use_mcmc and hasattr(posterior, '_sample_with'):
        original_sample_with = posterior._sample_with
        print("[maybe_force_mcmc] Posterior supports MCMC. Forcing 'mcmc' mode...")
        posterior._sample_with = 'mcmc'
        try:
            yield
        finally:
            posterior._sample_with = original_sample_with
            print("[maybe_force_mcmc] Restored original sampling method.")
    else:
        if use_mcmc:
            print("[maybe_force_mcmc] MCMC not supported for this posterior type.")
            print("[maybe_force_mcmc] Proceeding with direct neural sampling.")
        yield


def run_sbc(
        sbi_model,
        num_tests: int = 100,
        num_posterior_samples: int = 1000,
        save_path: Optional[str] = None,
        use_mcmc: bool = True
    ):
        """
        Run Simulation-Based Calibration test.
        """
        print(f"\nRunning SBC with {num_tests} tests...")
        if use_mcmc:
            print("Using MCMC sampling (slower but more robust)")

        ranks = []
        param_names = ['drift_rate', 'reward_bump', 'failure_bump']

        successful_tests = 0
        attempts = 0
        max_attempts = num_tests * 3  # Allow some failures
        pbar = tqdm(total=num_tests, desc="SBC tests")

        # 👇 Wrap the full SBC loop in the context manager
        with maybe_force_mcmc(sbi_model.posterior, use_mcmc=use_mcmc):
            while successful_tests < num_tests and attempts < max_attempts:
                attempts += 1
                try:
                    # Sample true theta from prior
                    theta_true = sbi_model.prior.sample()

                    # Simulate data
                    x_obs = sbi_model.simulate_for_sbi(theta_true)

                    # Skip invalid samples
                    if torch.isnan(x_obs).any():
                        continue

                    # Sample posterior
                    posterior_samples = sbi_model.infer(
                        x_obs,
                        num_samples=num_posterior_samples,
                        show_progress=False
                    )

                    if torch.isnan(posterior_samples).any():
                        continue

                    # Compute SBC rank statistic
                    rank = (posterior_samples < theta_true).sum(dim=0).numpy()
                    ranks.append(rank)
                    successful_tests += 1
                    pbar.update(1)

                except Exception:
                    continue

        pbar.close()

        if successful_tests < num_tests:
            print(f"\nWarning: Only {successful_tests}/{num_tests} tests succeeded")

        ranks = np.array(ranks)

        # --- Plot and diagnostics (unchanged) ---
        results = {
            'ranks': ranks,
            'num_tests': num_tests,
            'num_posterior_samples': num_posterior_samples,
        }

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for i, (ax, name) in enumerate(zip(axes, param_names)):
            ax.hist(ranks[:, i], bins=20, density=False, alpha=0.7,
                    edgecolor='black', label='Observed')
            expected_count = num_tests / 20
            ax.axhline(expected_count, color='red', linestyle='--',
                    linewidth=2, label='Expected (uniform)')
            se = np.sqrt(expected_count * (1 - 1/20))
            ax.axhline(expected_count + 2*se, color='red', linestyle=':', alpha=0.5)
            ax.axhline(expected_count - 2*se, color='red', linestyle=':', alpha=0.5)
            ax.set_xlabel(f'Rank of true {name}')
            ax.set_ylabel('Count')
            ax.set_title(f'SBC: {name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"SBC plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

        print("\n" + "="*60)
        print("SBC RESULTS")
        print("="*60)
        print("If posteriors are well-calibrated, ranks should be uniform.")
        print("Look for:")
        print("  - Flat histograms (good)")
        print("  - U-shaped (posterior too narrow)")
        print("  - Peaked (posterior too wide)")
        print("\nRank statistics (should be ~500 if uniform):")
        for i, name in enumerate(param_names):
            mean_rank = ranks[:, i].mean()
            std_rank = ranks[:, i].std()
            print(f"  {name:15s}: mean={mean_rank:6.1f}, std={std_rank:5.1f}")
        print("="*60 + "\n")

        return results

def plot_pairplot(
    posterior_samples: torch.Tensor,
    true_theta: Optional[torch.Tensor] = None,
    prior_samples: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None
):
    """
    Create pairplot showing posterior correlations.
    
    Args:
        posterior_samples: (n_samples, 3) posterior samples
        true_theta: (3,) true parameter values (optional)
        prior_samples: (n_samples, 3) prior samples for comparison (optional)
        save_path: Where to save plot
    """
    samples = posterior_samples.cpu().numpy()
    param_names = ['drift_rate', 'reward_bump', 'failure_bump']
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: marginal distributions
                ax.hist(samples[:, i], bins=30, density=True, alpha=0.7,
                       color='blue', edgecolor='black', label='Posterior')
                
                if prior_samples is not None:
                    prior_np = prior_samples.cpu().numpy()
                    ax.hist(prior_np[:, i], bins=30, density=True, alpha=0.3,
                           color='gray', edgecolor='black', label='Prior')
                
                if true_theta is not None:
                    ax.axvline(true_theta[i].item(), color='red', 
                              linestyle='--', linewidth=2, label='True')
                
                ax.set_ylabel('Density')
                if i == 0:
                    ax.legend()
                
            elif i > j:
                # Lower triangle: 2D scatter plots
                ax.scatter(samples[:, j], samples[:, i], alpha=0.3, s=1)
                
                if true_theta is not None:
                    ax.scatter(true_theta[j].item(), true_theta[i].item(),
                             color='red', s=100, marker='*', 
                             edgecolor='black', linewidth=1, zorder=10)
                
                # Compute correlation
                corr = np.corrcoef(samples[:, j], samples[:, i])[0, 1]
                ax.text(0.05, 0.95, f'r={corr:.2f}', 
                       transform=ax.transAxes, 
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
    
    # Print correlation matrix
    print("\nPosterior Correlation Matrix:")
    print("="*60)
    corr_matrix = np.corrcoef(samples.T)
    print("                ", "  ".join([f"{name:12s}" for name in param_names]))
    for i, name in enumerate(param_names):
        print(f"{name:15s}", "  ".join([f"{corr_matrix[i,j]:12.3f}" for j in range(3)]))
    print("="*60 + "\n")
    
    # Check for high correlations
    high_corr = []
    for i in range(3):
        for j in range(i+1, 3):
            if abs(corr_matrix[i, j]) > 0.7:
                high_corr.append((param_names[i], param_names[j], corr_matrix[i, j]))
    
    if high_corr:
        print("⚠️  WARNING: High correlations detected!")
        print("These parameters may not be well-identified:")
        for name1, name2, corr in high_corr:
            print(f"  {name1} ↔ {name2}: r = {corr:.3f}")
        print("\nConsider:")
        print("  1. Fixing one of the correlated parameters")
        print("  2. Collecting more informative data")
        print("  3. Adding informative priors")
        print()


def compare_prior_posterior(
    sbi_model,
    x_obs: torch.Tensor,
    num_samples: int = 5000,
    save_path: Optional[str] = None
):
    """
    Compare prior and posterior distributions.
    
    Helps identify which parameters are constrained by data.
    
    Args:
        sbi_model: Trained SBIInference instance
        x_obs: Observed data window
        num_samples: Number of samples to draw
        save_path: Where to save plot
    """
    # Sample from prior
    prior_samples = sbi_model.prior.sample((num_samples,))
    
    # Get posterior
    posterior_samples = sbi_model.infer(x_obs, num_samples=num_samples)
    
    # Plot
    param_names = ['drift_rate', 'reward_bump', 'failure_bump']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        # Prior
        ax.hist(prior_samples[:, i].numpy(), bins=30, density=True, 
               alpha=0.4, color='gray', edgecolor='black', label='Prior')
        
        # Posterior
        ax.hist(posterior_samples[:, i].numpy(), bins=30, density=True,
               alpha=0.7, color='blue', edgecolor='black', label='Posterior')
        
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.set_title(f'Prior vs Posterior: {name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Compute KL divergence approximation (how much info gained)
        # Using variance ratio as simple metric
        prior_std = prior_samples[:, i].std().item()
        post_std = posterior_samples[:, i].std().item()
        reduction = (1 - post_std / prior_std) * 100
        
        ax.text(0.05, 0.95, f'Variance reduction: {reduction:.1f}%',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Prior vs posterior plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


# Test the diagnostics
if __name__ == "__main__":
    print("Testing diagnostic tools...\n")
    
    from sbi_inference import SBIInference
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator_archive import PatchForagingDDM, create_prior
    
    # Train a small model for testing
    print("Training small model for testing...")
    simulator = PatchForagingDDM()
    sbi_model = SBIInference(simulator, create_prior(), device='cpu')
    sbi_model.train(num_simulations=5000, batch_size=50)
    
    # Test 1: SBC
    print("\n" + "="*60)
    print("Test 1: Simulation-Based Calibration")
    print("="*60)
    sbc_results = run_sbc(sbi_model, num_tests=50, save_path='sbc_diagnostic.png')
    
    # Test 2: Pairplot
    print("\n" + "="*60)
    print("Test 2: Pairplot Analysis")
    print("="*60)
    
    # Generate test data
    test_theta = torch.tensor([0.5, 0.6, 0.2])
    test_window, _, _ = simulator.simulate_window_with_drift(
        theta_mean=test_theta, window_sites=100, drift_sigma=0.05
    )
    
    posterior = sbi_model.infer(test_window, num_samples=2000)
    prior_samples = sbi_model.prior.sample((2000,))
    
    plot_pairplot(posterior, true_theta=test_theta, prior_samples=prior_samples,
                 save_path='pairplot_diagnostic.png')
    
    # Test 3: Prior vs Posterior
    print("\n" + "="*60)
    print("Test 3: Prior vs Posterior Comparison")
    print("="*60)
    compare_prior_posterior(sbi_model, test_window, save_path='prior_posterior.png')
    
    print("\nAll diagnostic tests complete!")