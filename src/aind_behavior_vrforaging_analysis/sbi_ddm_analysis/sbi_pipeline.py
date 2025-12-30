"""
Modular SBI pipeline for VR foraging DDM parameter inference
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import jax.numpy as jnp
from jax import random
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_inference_jax import train_snle, infer_parameters_snle


# ============================================================================
# 1. TRAIN MODEL
# ============================================================================

def train_snle_model(simulator, prior_fn, config, save_path=None):
    """
    Train SNLE model once - can be reused for all inference.
    
    Args:
        simulator: PatchForagingDDM_JAX instance
        prior_fn: Prior distribution function
        config: Dictionary with training settings
        save_path: Path to save trained model (optional)
    
    Returns:
        model_data: Dict with trained model, params, and normalization stats
    """
    print(f"\n{'='*70}")
    print(f"TRAINING SNLE MODEL")
    print(f"{'='*70}")
    
    rng_key = random.PRNGKey(config.get('seed', 42))
    
    snle, snle_params, losses, rng_key, y_mean, y_std = train_snle(
        simulator,
        prior_fn,
        mode='multi',
        n_simulations=config['n_simulations'],
        n_iter=config['n_iter'],
        batch_size=config['batch_size'],
        n_early_stopping_patience=config['n_early_stopping_patience'],
        learning_rate=config['learning_rate'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        rng_key=rng_key
    )
    
    model_data = {
        'snle': snle,
        'snle_params': snle_params,
        'losses': losses,
        'y_mean': y_mean,
        'y_std': y_std,
        'config': config
    }
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to: {save_path}")
    
    print(f"Training complete. Final train loss: {losses['train_losses'][-1]:.4f}")
    
    return model_data


# ============================================================================
# 2. VALIDATE ON SIMULATED DATA
# ============================================================================

def validate_parameter_recovery(model_data, simulator, prior_fn, n_test=50, seed=123):
    """
    Test if trained model can recover known parameters from simulated data.
    
    Args:
        model_data: Trained model from train_snle_model()
        simulator: Simulator instance
        prior_fn: Prior function
        n_test: Number of test cases
        seed: Random seed
    
    Returns:
        results: Dict with true params, inferred params, and recovery metrics
    """
    print(f"\n{'='*70}")
    print(f"VALIDATING PARAMETER RECOVERY")
    print(f"{'='*70}")
    
    rng_key = random.PRNGKey(seed)
    param_names = ['drift_rate', 'reward_bump', 'failure_bump', 'noise_std']
    
    true_params = []
    inferred_params = []
    
    for i in range(n_test):
        if (i + 1) % 10 == 0:
            print(f"  Test {i+1}/{n_test}")
        
        # Sample true parameters
        rng_key, subkey = random.split(rng_key)
        true_theta = prior_fn().sample(seed=subkey)['theta']
        true_params.append(np.array(true_theta))
        
        # Simulate observation
        rng_key, subkey = random.split(rng_key)
        _, observed_stats = simulator.simulate_one_window(true_theta, subkey)
        
        # Infer parameters
        rng_key, subkey = random.split(rng_key)
        posterior_samples, rng_key = infer_parameters_snle(
            model_data['snle'],
            model_data['snle_params'],
            observed_stats,
            model_data['y_mean'],
            model_data['y_std'],
            num_samples=500,
            num_warmup=200,
            num_chains=2,
            rng_key=subkey
        )
        
        # Use posterior mean as point estimate
        inferred_theta = jnp.mean(posterior_samples, axis=0)
        inferred_params.append(np.array(inferred_theta))
    
    true_params = np.array(true_params)
    inferred_params = np.array(inferred_params)
    
    # Compute recovery metrics
    print(f"\nParameter Recovery Results:")
    print(f"{'Parameter':<15s} | {'R²':>8s} | {'Mean Error':>12s}")
    print(f"{'-'*15} | {'-'*8} | {'-'*12}")
    
    for i, name in enumerate(param_names):
        # R² score
        ss_res = np.sum((true_params[:, i] - inferred_params[:, i])**2)
        ss_tot = np.sum((true_params[:, i] - true_params[:, i].mean())**2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Mean absolute error
        mae = np.mean(np.abs(true_params[:, i] - inferred_params[:, i]))
        
        print(f"{name:<15s} | {r2:>8.3f} | {mae:>12.4f}")
    
    # Plot recovery
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, name in enumerate(param_names):
        ax = axes[i]
        ax.scatter(true_params[:, i], inferred_params[:, i], alpha=0.5, s=30)
        
        # Add diagonal line
        min_val = min(true_params[:, i].min(), inferred_params[:, i].min())
        max_val = max(true_params[:, i].max(), inferred_params[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect recovery')
        
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Inferred {name}')
        ax.set_title(f'{name} Recovery')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    results = {
        'true_params': true_params,
        'inferred_params': inferred_params,
        'param_names': param_names
    }
    
    return results, fig


# ============================================================================
# 3. INFER PARAMETERS FROM REAL DATA (PER WINDOW)
# ============================================================================

def infer_parameters_per_window(model_data, observed_windows, seed=456):
    """
    Infer parameters separately for each window to track temporal dynamics.
    
    Args:
        model_data: Trained model from train_snle_model()
        observed_windows: Array of shape (n_windows, window_size, 3)
        seed: Random seed
    
    Returns:
        posterior_samples_per_window: List of arrays, one per window
    """
    print(f"\n{'='*70}")
    print(f"INFERRING PARAMETERS PER WINDOW")
    print(f"{'='*70}")
    
    n_windows = len(observed_windows)
    print(f"Processing {n_windows} windows...")
    
    rng_key = random.PRNGKey(seed)
    posterior_samples_per_window = []
    
    for i in range(n_windows):
        if (i + 1) % 10 == 0:
            print(f"  Window {i+1}/{n_windows}")
        
        # Get summary stats for this window
        window_data = jnp.array(observed_windows[i])
        from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.enhanced_stats import compute_enhanced_summary_stats
        observed_stats = compute_enhanced_summary_stats(window_data)
        
        # Infer parameters
        rng_key, subkey = random.split(rng_key)
        posterior_samples, rng_key = infer_parameters_snle(
            model_data['snle'],
            model_data['snle_params'],
            observed_stats,
            model_data['y_mean'],
            model_data['y_std'],
            num_samples=500,
            num_warmup=200,
            num_chains=2,
            rng_key=subkey
        )
        
        posterior_samples_per_window.append(np.array(posterior_samples))
    
    print(f"Inference complete for {n_windows} windows")
    
    return posterior_samples_per_window


# ============================================================================
# 4. ANALYZE TEMPORAL DYNAMICS
# ============================================================================

def analyze_temporal_dynamics(posterior_samples_per_window, param_names, 
                              odor_name=None, save_path=None):
    """
    Analyze how parameters change over the session.
    
    Args:
        posterior_samples_per_window: List of posterior samples per window
        param_names: List of parameter names
        odor_name: Name for plot titles
        save_path: Path to save figure
    
    Returns:
        summary_stats: Dict with mean, std over time
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING TEMPORAL DYNAMICS")
    if odor_name:
        print(f"Odor: {odor_name}")
    print(f"{'='*70}")
    
    n_windows = len(posterior_samples_per_window)
    n_params = len(param_names)
    
    # Compute posterior means and credible intervals per window
    means = np.zeros((n_windows, n_params))
    ci_low = np.zeros((n_windows, n_params))
    ci_high = np.zeros((n_windows, n_params))
    
    for i, samples in enumerate(posterior_samples_per_window):
        means[i] = np.mean(samples, axis=0)
        ci_low[i] = np.percentile(samples, 5, axis=0)
        ci_high[i] = np.percentile(samples, 95, axis=0)
    
    # Plot temporal dynamics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    window_indices = np.arange(n_windows)
    
    for i, name in enumerate(param_names):
        ax = axes[i]
        
        ax.plot(window_indices, means[:, i], 'o-', linewidth=2, markersize=4, label='Posterior mean')
        ax.fill_between(window_indices, ci_low[:, i], ci_high[:, i], alpha=0.3, label='90% CI')
        
        ax.set_xlabel('Window index (time →)')
        ax.set_ylabel(name)
        ax.set_title(f'{name} over session' + (f' ({odor_name})' if odor_name else ''))
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved temporal dynamics plot to: {save_path}")
    
    summary_stats = {
        'means': means,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'window_indices': window_indices
    }
    
    return summary_stats, fig


# ============================================================================
# 5. COMPARE ACROSS ODORS
# ============================================================================

def compare_odors_temporal(results_by_odor, param_names, save_path=None):
    """
    Compare temporal dynamics across odor types.
    
    Args:
        results_by_odor: Dict mapping odor_type -> summary_stats from analyze_temporal_dynamics
        param_names: List of parameter names
        save_path: Path to save figure
    """
    print(f"\n{'='*70}")
    print(f"COMPARING TEMPORAL DYNAMICS ACROSS ODORS")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = {'Methyl_Butyrate': 'blue', 'Alpha_pinene': 'orange'}
    
    for i, name in enumerate(param_names):
        ax = axes[i]
        
        for odor_type, stats in results_by_odor.items():
            color = colors.get(odor_type, 'gray')
            
            ax.plot(stats['window_indices'], stats['means'][:, i], 
                   'o-', linewidth=2, markersize=3, label=odor_type, 
                   color=color, alpha=0.7)
            ax.fill_between(stats['window_indices'], 
                           stats['ci_low'][:, i], stats['ci_high'][:, i], 
                           alpha=0.2, color=color)
        
        ax.set_xlabel('Window index (time →)')
        ax.set_ylabel(name)
        ax.set_title(f'{name} comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to: {save_path}")
    
    return fig