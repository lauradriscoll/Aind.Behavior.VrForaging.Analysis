"""
Utilities for SNLE analysis and visualization (JAX version).

Purpose: Plotting, diagnostics, and comparison functions for SNLE inference using sbijax.
"""

import os
# Force CPU backend on Apple Silicon to avoid Metal issues
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime

import arviz as az
import xarray as xr


def extract_samples(inference_results):
    """
    Robust extraction of posterior samples from sbijax's sample_posterior output.
    Supports InferenceData, Dataset, and tuples of both.
    """

    # Case 1: sbijax returned an InferenceData directly
    if isinstance(inference_results, az.InferenceData):
        posterior = inference_results.posterior
        # posterior is an xarray.Dataset
        return posterior

    # Case 2: tuple returned → unpack first element
    if isinstance(inference_results, tuple):
        first = inference_results[0]
        # InferenceData inside tuple
        if isinstance(first, az.InferenceData):
            return first.posterior
        # Dataset inside tuple
        if isinstance(first, xr.Dataset):
            return first

    # Case 3: plain Dataset
    if isinstance(inference_results, xr.Dataset):
        return inference_results

    raise TypeError(f"Could not extract posterior Dataset from type: {type(inference_results)}")


def plot_real_synth_hist(real_data,synthetic_data):

    var_names = [# Basic (7)
    "max_time", "mean_time", "std_time", 
    "mean_stops", "std_stops", "mean_rewards", "std_rewards",
    
    # Reward history (4) - KEY FEATURES
    "mean_time_after_reward",    # Should be affected by reward_bump
    "mean_time_after_failure",   # Should be affected by failure_bump
    "std_time_after_reward",
    "std_time_after_failure",
    
    # Temporal (5)
    "early_mean", "late_mean", "temporal_trend",
    "late_minus_early", "middle_mean",
    
    # Distribution (4)
    "p25", "median", "p75", "iqr",
    
    # Sequential (3)
    "autocorr_lag1", "diff_std", "mean_abs_change",
    
    # Reward stats (3)
    "reward_rate", "mean_reward_trial", "prop_patches_with_reward",
    
    # Patch stats (3)
    "n_patches", "mean_sites_per_patch", "stop_rate",
    ]

    print(var_names)
    # Determine number of rows/cols for subplots
    n_vars = len(var_names)
    n_cols = 3
    n_rows = int(np.ceil(n_vars / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2.5, n_rows*2.5))
    axes = axes.flatten()

    # Function to clean up axis
    def clean_axis(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)

    # Loop over each variable and plot histograms
    for i in range(n_vars):

        bins = 10
        axes[i].hist(synthetic_data[:, i], bins=bins, alpha=0.6, label='SNLE', density=True)
        axes[i].hist(real_data[:, i], bins=bins, alpha=0.6, label='Simulator', density=True)

        axes[i].axvline(synthetic_data[:, i].mean(), color='C0', linestyle='--', label='SNLE Mean')
        axes[i].axvline(real_data[:, i].mean(), color='C1', linestyle='--', label='Simulator Mean')

        axes[i].set_title(var_names[i], fontsize=14)
        axes[i].set_xlabel('Value', fontsize=12)
        axes[i].set_ylabel('Density', fontsize=12)
        axes[i].legend(fontsize=10)
        clean_axis(axes[i])

    # Remove empty subplots if n_vars < n_rows * n_cols
    for j in range(n_vars, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.suptitle('SNLE vs Simulator: Distribution of All Patch Statistics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Print summary statistics for all variables
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    for i, name in enumerate(var_names):
        print(f"\n{name}:")
        print(f"  SNLE:      {synthetic_data[:, i].mean():.3f} ± {synthetic_data[:, i].std():.3f}")
        print(f"  Simulator: {real_data[:, i].mean():.3f} ± {real_data[:, i].std():.3f}")
    print("="*60)

    

def plot_posterior_pairplot(prior_low, prior_high, posterior_samples, true_theta, save_path=None):
    """
    Create pairplot comparing prior and posterior distributions.
    
    Args:
        prior_low: (4,) lower bounds of prior
        prior_high: (4,) upper bounds of prior
        posterior_samples: (N, 4) posterior samples (JAX array or numpy)
        true_theta: (4,) true parameter values (JAX array or numpy)
        save_path: Optional path to save figure
    """
    # Convert to numpy if needed
    if isinstance(posterior_samples, jnp.ndarray):
        posterior_samples = np.array(posterior_samples)
    if isinstance(true_theta, jnp.ndarray):
        true_theta = np.array(true_theta)
    
    # Generate prior samples for comparison
    prior_low_np = np.array(prior_low)
    prior_high_np = np.array(prior_high)
    prior_samples = np.random.uniform(prior_low_np, prior_high_np, size=(1000, 4))
    
    param_names = [r"drift_rate", r"reward_bump", r"failure_bump", r"noise_std"]
    n_params = 4
    
    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
    
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: plot marginal distributions
                ax.hist(prior_samples[:, i], bins=30, alpha=0.5, density=True, 
                       label='Prior', color='blue', edgecolor='black')
                ax.hist(posterior_samples[:, i], bins=30, alpha=0.5, density=True,
                       label='Posterior', color='orange', edgecolor='black')
                ax.axvline(true_theta[i], color='red', linestyle='--', linewidth=2,
                          label='True' if i == 0 else '')
                if i == 0:
                    ax.legend(fontsize=8)
                ax.set_ylabel('Density', fontsize=10)
                
            elif i > j:
                # Lower triangle: scatter plots
                ax.scatter(prior_samples[:, j], prior_samples[:, i], 
                          alpha=0.3, s=5, color='blue', label='Prior')
                ax.scatter(posterior_samples[:, j], posterior_samples[:, i],
                          alpha=0.3, s=5, color='orange', label='Posterior')
                ax.plot(true_theta[j], true_theta[i], 'r*', markersize=15,
                       label='True')
                
            else:
                # Upper triangle: hide
                ax.axis('off')
            
            # Labels
            if i == n_params - 1:
                ax.set_xlabel(param_names[j], fontsize=10)
            else:
                ax.set_xticklabels([])
            
            if j == 0 and i != j:
                ax.set_ylabel(param_names[i], fontsize=10)
            elif i != j:
                ax.set_yticklabels([])
            
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('SNLE: Prior vs Posterior', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Pairplot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_snle_vs_simulator(simulator, true_theta, posterior_samples, 
                              num_patches=100, rng_key=None, save_path=None):
    """
    Compare data generated from SNLE posterior vs true simulator.
    
    Args:
        simulator: PatchForagingDDM_JAX instance
        true_theta: True parameters (JAX array)
        posterior_samples: Posterior parameter samples (JAX array)
        num_patches: Number of patches to generate for comparison
        rng_key: JAX random key
        save_path: Optional path to save figure
    """
    if rng_key is None:
        rng_key = random.PRNGKey(0)
    
    print(f"\nGenerating {num_patches} windows from each model...")
    
    # Convert to numpy for easier handling
    true_theta_np = np.array(true_theta)
    posterior_samples_np = np.array(posterior_samples)
    
    # 1. Generate "real" data from true simulator
    real_stats = []
    for _ in range(num_patches):
        rng_key, subkey = random.split(rng_key)
        _, stats = simulator.simulate_one_window(true_theta, subkey)
        real_stats.append(np.array(stats))
    real_data = np.array(real_stats)
    
    # 2. Generate "synthetic" data from posterior-sampled parameters
    synthetic_stats = []
    for i in range(num_patches):
        theta_sample = posterior_samples[i % len(posterior_samples)]
        rng_key, subkey = random.split(rng_key)
        _, stats = simulator.simulate_one_window(theta_sample, subkey)
        synthetic_stats.append(np.array(stats))
    synthetic_data = np.array(synthetic_stats)
    
    # Extract relevant statistics (assuming multi-patch mode with 7 stats)
    # [max_time, mean_time, std_time, mean_stops, std_stops, mean_rewards, std_rewards]
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    # Plot 1: Mean patch time distributions
    axes[0].hist(synthetic_data[:, 1], bins=30, alpha=0.7, label='SNLE', 
                 density=True, edgecolor='black', color='orange')
    axes[0].hist(real_data[:, 1], bins=30, alpha=0.7, label='Simulator', 
                 density=True, edgecolor='black', color='blue')
    axes[0].set_xlabel('Mean Patch Time')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Mean Patch Time Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Std patch time
    axes[1].hist(synthetic_data[:, 2], bins=30, alpha=0.7, label='SNLE',
                 density=True, edgecolor='black', color='orange')
    axes[1].hist(real_data[:, 2], bins=30, alpha=0.7, label='Simulator',
                 density=True, edgecolor='black', color='blue')
    axes[1].set_xlabel('Std Patch Time')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Patch Time Variability')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Mean stops
    axes[2].hist(synthetic_data[:, 3], bins=30, alpha=0.7, label='SNLE',
                 density=True, edgecolor='black', color='orange')
    axes[2].hist(real_data[:, 3], bins=30, alpha=0.7, label='Simulator',
                 density=True, edgecolor='black', color='blue')
    axes[2].set_xlabel('Mean Stops per Window')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Mean Stops Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Mean rewards
    axes[3].hist(synthetic_data[:, 5], bins=30, alpha=0.7, label='SNLE',
                 density=True, edgecolor='black', color='orange')
    axes[3].hist(real_data[:, 5], bins=30, alpha=0.7, label='Simulator',
                 density=True, edgecolor='black', color='blue')
    axes[3].set_xlabel('Mean Rewards per Window')
    axes[3].set_ylabel('Density')
    axes[3].set_title('Mean Rewards Distribution')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # Plot 5: Summary comparison
    metrics = ['Mean Time', 'Std Time', 'Mean Stops', 'Std Stops', 'Mean Rewards', 'Std Rewards']
    snle_means = [synthetic_data[:, i].mean() for i in [1, 2, 3, 4, 5, 6]]
    sim_means = [real_data[:, i].mean() for i in [1, 2, 3, 4, 5, 6]]
    
    x = np.arange(len(metrics))
    width = 0.35
    axes[4].bar(x - width/2, snle_means, width, label='SNLE', alpha=0.7, 
               edgecolor='black', color='orange')
    axes[4].bar(x + width/2, sim_means, width, label='Simulator', alpha=0.7,
               edgecolor='black', color='blue')
    axes[4].set_ylabel('Mean Value')
    axes[4].set_title('Summary Statistics Comparison')
    axes[4].set_xticks(x)
    axes[4].set_xticklabels(metrics, rotation=45, ha='right')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Max time comparison
    axes[5].hist(synthetic_data[:, 0], bins=30, alpha=0.7, label='SNLE',
                 density=True, edgecolor='black', color='orange')
    axes[5].hist(real_data[:, 0], bins=30, alpha=0.7, label='Simulator',
                 density=True, edgecolor='black', color='blue')
    axes[5].set_xlabel('Max Patch Time')
    axes[5].set_ylabel('Density')
    axes[5].set_title('Max Patch Time Distribution')
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)
    
    plt.suptitle('SNLE vs Simulator: Summary Statistics Comparison', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    for i, metric in enumerate(['Max Time', 'Mean Time', 'Std Time', 'Mean Stops', 
                                 'Std Stops', 'Mean Rewards', 'Std Rewards']):
        print(f"\n{metric}:")
        print(f"  SNLE:      {synthetic_data[:, i].mean():.3f} ± {synthetic_data[:, i].std():.3f}")
        print(f"  Simulator: {real_data[:, i].mean():.3f} ± {real_data[:, i].std():.3f}")
    print(f"{'='*60}")


def plot_posterior_distributions(posterior_samples, true_theta=None, save_path=None):
    """
    Plot marginal posterior distributions for each parameter.
    
    Args:
        posterior_samples: (N, 4) posterior samples (JAX or numpy array)
        true_theta: Optional true parameter values (JAX or numpy array)
        save_path: Optional path to save figure
    """
    # Convert to numpy if needed
    if isinstance(posterior_samples, jnp.ndarray):
        samples_np = np.array(posterior_samples)
    else:
        samples_np = posterior_samples
    
    if true_theta is not None and isinstance(true_theta, jnp.ndarray):
        true_theta = np.array(true_theta)
    
    param_names = ['drift_rate', 'reward_bump', 'failure_bump', 'noise_std']
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(samples_np[:, i], bins=50, density=True, alpha=0.7, 
               edgecolor='black', color='blue')
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.set_title(f'Posterior: {name}')
        
        if true_theta is not None:
            ax.axvline(true_theta[i], color='red', linestyle='--', 
                      linewidth=2, label='True value')
            ax.legend()
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Posterior distributions saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_inference_summary(posterior_samples, true_theta):
    """
    Print summary statistics of inference results.
    
    Args:
        posterior_samples: (N, 4) posterior samples (JAX or numpy array)
        true_theta: (4,) true parameter values (JAX or numpy array)
    """
    # Convert to numpy for easier handling
    if isinstance(posterior_samples, jnp.ndarray):
        posterior_samples = np.array(posterior_samples)
    if isinstance(true_theta, jnp.ndarray):
        true_theta = np.array(true_theta)
    
    posterior_mean = posterior_samples.mean(axis=0)
    posterior_std = posterior_samples.std(axis=0)
    absolute_error = np.abs(posterior_mean - true_theta)
    relative_error = absolute_error / (np.abs(true_theta) + 1e-6)
    
    param_names = ['drift_rate', 'reward_bump', 'failure_bump', 'noise_std']
    
    print(f"\n{'='*60}")
    print("INFERENCE RESULTS")
    print(f"{'='*60}")
    print(f"\n{'Parameter':15s} {'True':>10s} {'Mean':>10s} {'Std':>10s} {'AbsErr':>10s} {'RelErr':>10s}")
    print("-" * 60)
    
    for i, name in enumerate(param_names):
        print(f"{name:15s} {true_theta[i]:10.4f} "
              f"{posterior_mean[i]:10.4f} {posterior_std[i]:10.4f} "
              f"{absolute_error[i]:10.4f} {relative_error[i]:10.2%}")
    
    print(f"{'='*60}")
    
    # Overall assessment
    mean_abs_error = absolute_error.mean()
    mean_rel_error = relative_error.mean()
    
    print(f"\nOverall Performance:")
    print(f"  Mean absolute error: {mean_abs_error:.4f}")
    print(f"  Mean relative error: {mean_rel_error:.2%}")
    
    if mean_abs_error < 0.1:
        print("\n✓ Excellent parameter recovery!")
    elif mean_abs_error < 0.3:
        print("\n✓ Good parameter recovery")
    else:
        print("\n⚠️  Poor parameter recovery - consider more training data or better features")
    
    print(f"{'='*60}\n")


def save_snle_model(snle_params, y_mean, y_std, mode='multi', base_dir='snle_models'):
    """
    Save trained SNLE model and normalization parameters in timestamped folder.
    
    Args:
        snle: Trained SNLE object from sbijax
        snle_params: Trained parameters dict
        mode: 'single' or 'multi' for folder naming
        base_dir: Base directory for all models
    
    Returns:
        model_dir: Path to the created model directory
    """
    
    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(base_dir, f'{mode}_patch_{timestamp}')
    analysis_dir = os.path.join(model_dir, 'analysis')
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    print(f"\nSaving SNLE model to: {model_dir}")
    # Save parameters using pickle (JAX params are pytrees)
    model_dict = {
        'params': snle_params,
        'mode': mode,
        'timestamp': timestamp,
        'y_mean': y_mean,
        'y_std': y_std
    }
    
    params_path = os.path.join(model_dir, 'params.pkl')
    with open(params_path, 'wb') as f:
        pickle.dump(model_dict, f)
    
    # Note: We can't easily pickle the SNLE object itself due to TFP distributions
    # Users will need to reconstruct it with the same prior/simulator
    
    print(f"SNLE model saved to: {model_dir}")
    print(f"  - Parameters: {params_path}")
    print(f"  - Analysis folder: {analysis_dir}")
    
    return model_dir


def load_snle_model(model_dir):
    """
    Load trained SNLE parameters and normalization from timestamped folder.
    
    Note: This loads parameters only. You need to recreate the SNLE object
    with the same prior and simulator, then use these parameters.
    
    Args:
        model_dir: Path to model directory (e.g., 'snle_models/multi_patch_20241107_143022')
    
    Returns:
        snle_params: Trained parameters dict
        mode: 'single' or 'multi'
        analysis_dir: Path to analysis folder
    """
    
    params_path = os.path.join(model_dir, 'params.pkl')
    analysis_dir = os.path.join(model_dir, 'analysis')
    
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameters not found at {params_path}")
    
    with open(params_path, 'rb') as f:
        model_dict = pickle.load(f)
    
    print(f"SNLE parameters loaded from: {model_dir}")
    print(f"  - Mode: {model_dict['mode']}")
    print(f"  - Timestamp: {model_dict['timestamp']}")
    print(f"  - Analysis folder: {analysis_dir}")
    print(f"\n⚠️  Remember to recreate SNLE object with same prior/simulator")
    
    return (model_dict, analysis_dir)


# # Test
# if __name__ == "__main__":

#     from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM_JAX, create_prior_jax
#     from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.archive.snle_inference import infer_parameters_snle

#     model_dir = 'snle_models/multi_patch_20241107_143022'  # Example path
#     snle_params, mode, analysis_dir = load_snle_model(model_dir)
#     print("Testing SNLE utilities (JAX version)...")
    
#     true_theta = jnp.array([0.6, 0.8, 0.3, 0.05])
#     rng = random.PRNGKey(0)

#     rng_key, test_key = random.split(rng_key)
#     test_theta = prior_fn().sample(seed=test_key)
#     test_x = simulator.simulator_fn(test_theta, test_key)

#     # Create MAF with appropriate dimension
#     n_dim_data = test_x.shape[-1]  # should be 7
#     neural_network = make_maf(n_dimension=n_dim_data)
    
#     # --- 4. Create SNLE model ---
#     fns = prior_fn, simulator.simulator_fn
#     snle = NLE(fns, neural_network) #oddly this is what SNLE is called in sbijax - https://sbijax.readthedocs.io/en/latest/sbijax.html#sbijax.NLE
#     snle_params = snle_params

#     posterior_samples, rng_key = infer_parameters_snle(
#         snle,
#         snle_params,
#         observed_stats,
#         num_samples=100,  # Small for testing
#         num_warmup=50,
#         num_chains=2,
#         rng_key=subkey
#     )
    
#     print("\n1. Testing training history plot...")
#     plot_training_history(training_history, mode='multi')
    
#     print("\n2. Testing posterior distributions plot...")
#     plot_posterior_distributions(posterior_samples, true_theta)
    
#     print("\n3. Testing inference summary...")
#     print_inference_summary(posterior_samples, true_theta)
    
#     print("\n✓ All utility tests passed!")