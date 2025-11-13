"""
Parameter sweep analysis for SNLE: How do num_simulations and window_sites 
affect posterior accuracy and generative quality?

Purpose: Systematic evaluation of SNLE training parameters on inference quality.

Updates:
- Uses JAX simulator for fast training data generation
- Supports 4D theta [drift_rate, reward_bump, failure_bump, noise_std]
- Updated to use new evolve_params API
"""

# CRITICAL: Set JAX platform BEFORE any imports
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import torch
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import pickle
from jax import random

# Import SNLE modules
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM, create_prior
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator_jax import PatchForagingDDM_JAX
from snle_inference import train_snle, infer_parameters_snle
from snle_utils import plot_training_history

# Parameter grid
NUM_SIMULATIONS = [100000, 500000, 1000000]  # Use large numbers with JAX!
WINDOW_SITES = [25, 50, 75, 100]

# Test cases (4D theta: drift_rate, reward_bump, failure_bump, noise_std)
TEST_CASES = [
    ("low_drift_high_bump", torch.tensor([0.2, 0.8, 0.3, 0.05])),
    ("high_drift_low_bump", torch.tensor([0.8, 0.2, 0.3, 0.05])),
    ("balanced", torch.tensor([0.5, 0.5, 0.3, 0.05])),
    ("high_noise", torch.tensor([0.5, 0.5, 0.3, 0.1])),
]


def setup_logging(results_dir):
    """Setup logging to both console and file."""
    log_file = os.path.join(results_dir, 'sweep_log.txt')
    
    # Create logger
    logger = logging.getLogger('parameter_sweep')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def compute_kl_divergence_hist(true_samples, posterior_samples, num_bins=30):
    """
    Compute KL divergence between two distributions using histogram approximation.
    KL(P || Q) where P is true and Q is posterior
    """
    # Determine common bin edges
    all_data = np.concatenate([true_samples, posterior_samples])
    bin_edges = np.histogram_bin_edges(all_data, bins=num_bins)
    
    # Compute histograms (probabilities)
    p, _ = np.histogram(true_samples, bins=bin_edges, density=True)
    q, _ = np.histogram(posterior_samples, bins=bin_edges, density=True)
    
    # Normalize to sum to 1
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)
    
    # Add small epsilon to avoid log(0)
    p = p + 1e-10
    q = q + 1e-10
    
    # Compute KL divergence
    kl = np.sum(kl_div(p, q))
    
    return kl


def compute_generative_metrics(simulator, true_theta, posterior_samples, logger, num_patches=100):
    """
    Compare distributions of data generated from true params vs posterior samples.
    
    Uses JAX simulator for fast batch generation.
    Computes Wasserstein distance and KL divergence for each summary statistic.
    """
    logger.info(f"  Computing generative metrics ({num_patches} patches)...")
    
    # Use JAX simulator for fast batch generation
    import jax.numpy as jnp
    simulator_jax = PatchForagingDDM_JAX()
    rng_key = random.PRNGKey(0)
    
    # Generate from true parameters (batch)
    rng_key, subkey = random.split(rng_key)
    true_theta_batch = jnp.tile(jnp.array(true_theta.numpy()), (num_patches, 1))
    _, _, true_stats = simulator_jax.simulate_batch(true_theta_batch, subkey, return_aggregate=False)
    true_data = np.array(true_stats)
    
    # Generate from posterior samples (batch)
    rng_key, subkey = random.split(rng_key)
    posterior_theta_batch = jnp.array(posterior_samples[:num_patches].numpy())
    _, _, posterior_stats = simulator_jax.simulate_batch(posterior_theta_batch, subkey, return_aggregate=False)
    posterior_data = np.array(posterior_stats)
    
    # Compute Wasserstein distance and KL divergence for each feature
    # Features: [total_time, num_stops, num_rewards]
    feature_names = ['total_time', 'num_stops', 'num_rewards']
    wasserstein_distances = []
    kl_divergences = []
    
    for i, name in enumerate(feature_names):
        # Wasserstein distance
        wd = wasserstein_distance(true_data[:, i], posterior_data[:, i])
        wasserstein_distances.append(wd)
        
        # KL divergence (using histogram approximation)
        kl = compute_kl_divergence_hist(true_data[:, i], posterior_data[:, i])
        kl_divergences.append(kl)
        
        logger.info(f"    {name}: Wasserstein={wd:.4f}, KL={kl:.4f}")
    
    # Also compute mean/std differences
    mean_diff = np.abs(true_data.mean(axis=0) - posterior_data.mean(axis=0))
    std_diff = np.abs(true_data.std(axis=0) - posterior_data.std(axis=0))
    
    return {
        'wasserstein_total_time': wasserstein_distances[0],
        'wasserstein_num_stops': wasserstein_distances[1],
        'wasserstein_num_rewards': wasserstein_distances[2],
        'mean_wasserstein': np.mean(wasserstein_distances),
        'kl_total_time': kl_divergences[0],
        'kl_num_stops': kl_divergences[1],
        'kl_num_rewards': kl_divergences[2],
        'mean_kl': np.mean(kl_divergences),
        'mean_diff_total_time': mean_diff[0],
        'mean_diff_num_stops': mean_diff[1],
        'mean_diff_num_rewards': mean_diff[2],
        'std_diff_total_time': std_diff[0],
        'std_diff_num_stops': std_diff[1],
        'std_diff_num_rewards': std_diff[2],
    }


def evaluate_on_test_case(simulator, inference, x_mean, x_std, 
                          true_theta, test_name, logger):
    """
    Evaluate trained model on a single test case.
    
    Computes:
    - Posterior accuracy metrics
    - Generative quality metrics
    
    Returns:
        dict with all metrics
    """
    # 1. Generate observed data using PyTorch simulator
    param_gen = simulator.evolve_params(mode='walk', theta_init=true_theta, sigma=0.0)
    _, observed_stats, _ = simulator.simulate_trial(
        param_gen, 
        window_sites=100,  # Fixed for testing
        return_aggregate=False  # Use single-patch stats
    )
    
    # 2. Infer parameters
    posterior_samples = infer_parameters_snle(
        inference, observed_stats,
        x_mean, x_std,
        num_samples=1000,
        warmup_steps=200
    )
    
    # 3. Compute posterior accuracy metrics
    posterior_mean = posterior_samples.mean(dim=0)
    posterior_std = posterior_samples.std(dim=0)
    
    # Mean absolute error per parameter
    mae_per_param = (posterior_mean - true_theta).abs()
    
    # Overall RMSE
    rmse = torch.sqrt(((posterior_mean - true_theta) ** 2).mean())
    
    # Coverage: does true value fall in 95% credible interval?
    lower = torch.quantile(posterior_samples, 0.025, dim=0)
    upper = torch.quantile(posterior_samples, 0.975, dim=0)
    coverage = ((true_theta >= lower) & (true_theta <= upper)).float()
    
    # 4. Compute generative quality metrics
    gen_metrics = compute_generative_metrics(
        simulator, true_theta, posterior_samples, logger
    )
    
    metrics = {
        'test_name': test_name,
        'true_theta': true_theta.numpy(),
        'posterior_mean': posterior_mean.numpy(),
        'posterior_std': posterior_std.numpy(),
        'mae_per_param': mae_per_param.numpy(),
        'mean_mae': mae_per_param.mean().item(),
        'rmse': rmse.item(),
        'coverage': coverage.numpy(),
        'mean_coverage': coverage.mean().item(),
        **gen_metrics
    }
    
    logger.info(f"  MAE: {metrics['mean_mae']:.4f}, RMSE: {metrics['rmse']:.4f}, Coverage: {metrics['mean_coverage']:.2f}")
    
    return metrics


def save_intermediate_result(result, results_dir):
    """Save individual result as pickle and append summary to CSV."""
    if result is None:
        return
    
    # Save full result as pickle
    num_sims = result['num_simulations']
    window_sites = result['window_sites']
    pickle_path = os.path.join(results_dir, f'result_sims{num_sims}_sites{window_sites}.pkl')
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(result, f)
    
    # Append summary to CSV
    csv_path = os.path.join(results_dir, 'sweep_summary.csv')
    
    # Flatten results for CSV
    rows = []
    for test_result in result['test_results']:
        if test_result is not None:
            row = {
                'num_simulations': num_sims,
                'window_sites': window_sites,
                'final_train_loss': result['training_metrics']['final_train_loss'],
                'final_val_loss': result['training_metrics']['final_val_loss'],
                'best_val_loss': result['training_metrics']['best_val_loss'],
                'epochs_trained': result['training_metrics']['epochs_trained'],
                'test_case': test_result['test_name'],
                'mean_mae': test_result['mean_mae'],
                'rmse': test_result['rmse'],
                'mean_coverage': test_result['mean_coverage'],
                'mean_wasserstein': test_result['mean_wasserstein'],
                'mean_kl': test_result['mean_kl'],
            }
            rows.append(row)
    
    # Append to CSV
    df = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


def check_existing_model(num_sims, window_sites, results_dir):
    """
    Check if a model already exists for this parameter combination.
    
    Returns:
        (exists, model_dir, model_dict) tuple
    """
    model_dir = os.path.join(results_dir, f'model_sims{num_sims}_sites{window_sites}')
    model_path = os.path.join(model_dir, 'model.pkl')
    
    if os.path.exists(model_path):
        try:
            model_dict = torch.load(model_path)
            return True, model_dir, model_dict
        except Exception as e:
            # Model file corrupted, treat as not existing
            return False, model_dir, None
    
    return False, model_dir, None


def train_and_evaluate(num_sims, window_sites, results_dir, logger, 
                      rng_key, skip_if_exists=True, use_jax=True):
    """
    Train SNLE model and evaluate on all test cases.
    
    Args:
        num_sims: Number of training simulations
        window_sites: Number of sites per simulation (only used for PyTorch)
        results_dir: Directory to save results
        logger: Logger instance
        rng_key: JAX random key
        skip_if_exists: If True, skip training if model already exists
        use_jax: If True, use JAX simulator for fast data generation
    
    Returns:
        dict with all metrics for this parameter combination
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Training: num_simulations={num_sims}, window_sites={window_sites}")
    logger.info(f"{'='*80}")
    
    # Check if model already exists
    exists, model_dir, existing_model = check_existing_model(num_sims, window_sites, results_dir)
    
    if exists and skip_if_exists:
        logger.info(f"✓ Model already exists at: {model_dir}")
        logger.info(f"  Loading existing model and running evaluation only...")
        
        # Load existing model
        inference = existing_model['inference']
        x_mean = existing_model['x_mean']
        x_std = existing_model['x_std']
        
        # Check if we have training metrics
        if 'training_metrics' in existing_model:
            training_metrics = existing_model['training_metrics']
        else:
            training_metrics = {
                'final_train_loss': None,
                'final_val_loss': None,
                'best_val_loss': None,
                'epochs_trained': None,
            }
            logger.info("  (Training metrics not available for this model)")
    else:
        if exists:
            logger.info(f"⚠️  Model exists but skip_if_exists=False, retraining...")
        
        # Setup
        simulator = PatchForagingDDM_JAX() if use_jax else PatchForagingDDM()
        prior = create_prior()
        
        # Create model-specific directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Split RNG key for this training run
        rng_key, subkey = random.split(rng_key)
        
        # Train model
        try:
            logger.info(f"  Training SNLE (single mode, use_jax={use_jax})...")
            
            _, inference, x_mean, x_std, history = train_snle(
                simulator=simulator,
                prior=prior,
                num_simulations=num_sims,
                window_sites=window_sites,
                mode='single',
                max_num_epochs=50,
                batch_size=50,
                use_jax=use_jax,
                rng_key=subkey if use_jax else None
            )
            
            # Store training metrics
            training_metrics = {
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'best_val_loss': history['best_val_loss'],
                'epochs_trained': history['epochs_trained'],
            }
            
            logger.info(f"  Training complete: val_loss={training_metrics['final_val_loss']:.4f}")
            
            # Save training history plot
            plot_path = os.path.join(model_dir, 'training_history.png')
            plot_training_history(history, mode='single', save_path=plot_path)
            
            # Save the trained model
            model_save_path = os.path.join(model_dir, 'model.pkl')
            model_dict = {
                'inference': inference,
                'x_mean': x_mean,
                'x_std': x_std,
                'mode': 'single',
                'num_simulations': num_sims,
                'window_sites': window_sites,
                'training_metrics': training_metrics,
                'use_jax': use_jax,
            }
            torch.save(model_dict, model_save_path)
            logger.info(f"  Model saved to: {model_save_path}")
            
        except Exception as e:
            logger.error(f"  Training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, rng_key
    
    # Setup simulator for evaluation (use PyTorch for consistency)
    simulator = PatchForagingDDM()
    
    # Evaluate on all test cases
    test_results = []
    for test_name, true_theta in TEST_CASES:
        logger.info(f"\nEvaluating on test case: {test_name}")
        logger.info(f"  True theta: {true_theta}")
        
        try:
            metrics = evaluate_on_test_case(
                simulator, inference, x_mean, x_std,
                true_theta, test_name, logger
            )
            test_results.append(metrics)
            
        except Exception as e:
            logger.error(f"  Evaluation failed for {test_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            test_results.append(None)
    
    # Combine results
    result = {
        'num_simulations': num_sims,
        'window_sites': window_sites,
        'training_metrics': training_metrics,
        'test_results': test_results,
        'model_dir': model_dir,
        'use_jax': use_jax,
    }
    
    # Save intermediate result
    save_intermediate_result(result, results_dir)
    
    return result, rng_key


def create_summary_plots(results_dir, logger):
    """Create summary visualizations from sweep results."""
    logger.info("Creating summary plots...")
    
    # Load summary CSV
    csv_path = os.path.join(results_dir, 'sweep_summary.csv')
    if not os.path.exists(csv_path):
        logger.error("No summary CSV found!")
        return
    
    df = pd.read_csv(csv_path)
    
    # Average across test cases for each parameter combination
    df_avg = df.groupby(['num_simulations', 'window_sites']).agg({
        'mean_mae': 'mean',
        'rmse': 'mean',
        'mean_coverage': 'mean',
        'mean_wasserstein': 'mean',
        'mean_kl': 'mean',
        'final_val_loss': 'first',
    }).reset_index()
    
    # Create pivot tables for heatmaps
    metrics = ['mean_mae', 'rmse', 'mean_coverage', 'mean_wasserstein', 'mean_kl', 'final_val_loss']
    metric_labels = ['Mean MAE', 'RMSE', 'Coverage', 'Mean Wasserstein', 'Mean KL', 'Final Val Loss']
    
    # 1. Create heatmaps
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        pivot = df_avg.pivot(index='window_sites', columns='num_simulations', values=metric)
        
        im = axes[idx].imshow(pivot.values, aspect='auto', cmap='viridis', origin='lower')
        axes[idx].set_xticks(range(len(pivot.columns)))
        axes[idx].set_xticklabels([f'{int(x/1000)}K' for x in pivot.columns], rotation=45, ha='right')
        axes[idx].set_yticks(range(len(pivot.index)))
        axes[idx].set_yticklabels(pivot.index)
        axes[idx].set_xlabel('num_simulations', fontsize=10)
        axes[idx].set_ylabel('window_sites', fontsize=10)
        axes[idx].set_title(label, fontsize=12, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[idx])
        
        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                text = axes[idx].text(j, i, f'{pivot.values[i, j]:.3f}',
                                     ha="center", va="center", color="white", fontsize=8)
    
    plt.tight_layout()
    heatmap_path = os.path.join(results_dir, 'summary_heatmaps.png')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: summary_heatmaps.png")
    
    # 2. Line plots: Metrics vs num_sims
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        for window_sites in sorted(df_avg['window_sites'].unique()):
            data = df_avg[df_avg['window_sites'] == window_sites].sort_values('num_simulations')
            axes[idx].plot(data['num_simulations'], data[metric], 
                          marker='o', label=f'sites={window_sites}', linewidth=2)
        
        axes[idx].set_xlabel('num_simulations', fontsize=10)
        axes[idx].set_ylabel(label, fontsize=10)
        axes[idx].set_title(f'{label} vs num_simulations', fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xscale('log')
    
    plt.tight_layout()
    numsims_path = os.path.join(results_dir, 'metrics_vs_num_simulations.png')
    plt.savefig(numsims_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: metrics_vs_num_simulations.png")
    
    logger.info("Summary visualizations complete!")


def run_parameter_sweep(base_dir='snle_parameter_sweep', skip_if_exists=True, 
                       resume_dir=None, use_jax=True):
    """
    Run full parameter sweep experiment.
    
    Args:
        base_dir: Base directory for saving results (only used if resume_dir is None)
        skip_if_exists: If True, skip training for models that already exist
        resume_dir: If provided, resume from this existing results directory
        use_jax: If True, use JAX simulator for fast data generation
    """
    # Create or use existing results directory
    if resume_dir is not None:
        if not os.path.exists(resume_dir):
            raise ValueError(f"Resume directory does not exist: {resume_dir}")
        results_dir = resume_dir
        print(f"Resuming from existing directory: {results_dir}")
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(base_dir, f'sweep_{timestamp}')
        os.makedirs(results_dir, exist_ok=True)
        print(f"Starting new sweep in: {results_dir}")
    
    # Setup logging
    logger = setup_logging(results_dir)
    logger.info("="*80)
    logger.info("SNLE PARAMETER SWEEP ANALYSIS")
    logger.info("="*80)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"num_simulations: {NUM_SIMULATIONS}")
    logger.info(f"window_sites: {WINDOW_SITES}")
    logger.info(f"Test cases: {len(TEST_CASES)}")
    logger.info(f"Using JAX: {use_jax}")
    logger.info("="*80)
    
    # Initialize JAX random key
    rng_key = random.PRNGKey(42)
    
    # Initialize results storage
    all_results = []
    total_combinations = len(NUM_SIMULATIONS) * len(WINDOW_SITES)
    combination_idx = 0
    
    # Sweep over parameters
    num_skipped = 0
    num_trained = 0
    num_failed = 0
    
    for num_sims in NUM_SIMULATIONS:
        for window_sites in WINDOW_SITES:
            combination_idx += 1
            logger.info(f"\n\n{'#'*80}")
            logger.info(f"COMBINATION {combination_idx}/{total_combinations}")
            logger.info(f"{'#'*80}")
            
            # Check if already exists
            exists, _, _ = check_existing_model(num_sims, window_sites, results_dir)
            if exists and skip_if_exists:
                num_skipped += 1
            
            result, rng_key = train_and_evaluate(
                num_sims, window_sites, results_dir, logger, rng_key,
                skip_if_exists=skip_if_exists, use_jax=use_jax
            )
            
            if result is not None:
                all_results.append(result)
                if not (exists and skip_if_exists):
                    num_trained += 1
            else:
                num_failed += 1
            
            logger.info(f"\nCompleted {combination_idx}/{total_combinations} combinations")
            logger.info(f"  Trained: {num_trained}, Skipped: {num_skipped}, Failed: {num_failed}")
    
    # Generate summary visualizations
    logger.info("\n" + "="*80)
    logger.info("GENERATING SUMMARY VISUALIZATIONS")
    logger.info("="*80)
    
    create_summary_plots(results_dir, logger)
    
    logger.info("\n" + "="*80)
    logger.info("PARAMETER SWEEP COMPLETE!")
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"Successful combinations: {len(all_results)}/{total_combinations}")
    logger.info(f"  Newly trained: {num_trained}")
    logger.info(f"  Skipped (already existed): {num_skipped}")
    logger.info(f"  Failed: {num_failed}")
    logger.info("="*80)
    
    return results_dir, all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SNLE Parameter Sweep Analysis')
    parser.add_argument('mode', nargs='?', default='full', choices=['test', 'force', 'full'],
                       help='Run mode: test (quick test), force (retrain all), full (default, skip existing)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from existing results directory')
    parser.add_argument('--no-jax', action='store_true',
                       help='Use PyTorch simulator instead of JAX (slower)')
    
    args = parser.parse_args()
    
    use_jax = not args.no_jax
    
    if args.mode == 'test':
        # Quick test with reduced parameters
        print("Running quick test with reduced parameters...")
        NUM_SIMULATIONS = [10000, 50000]
        WINDOW_SITES = [25, 50]
        results_dir, all_results = run_parameter_sweep(
            base_dir='snle_parameter_sweep_test',
            resume_dir=args.resume,
            use_jax=use_jax
        )
    elif args.mode == 'force':
        # Force retraining even if models exist
        print("Running full sweep with forced retraining...")
        results_dir, all_results = run_parameter_sweep(
            skip_if_exists=False,
            resume_dir=args.resume,
            use_jax=use_jax
        )
    else:
        # Full parameter sweep (default: skip existing models)
        results_dir, all_results = run_parameter_sweep(
            resume_dir=args.resume,
            use_jax=use_jax
        )