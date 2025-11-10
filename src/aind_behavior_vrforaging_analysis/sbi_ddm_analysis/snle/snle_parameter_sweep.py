"""
Parameter sweep analysis for SNLE: How do num_simulations and window_sites 
affect posterior accuracy and generative quality?

Purpose: Systematic evaluation of SNLE training parameters on inference quality.
"""

import torch
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import logging
import pickle

# Import SNLE modules
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM, create_prior
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_inference import train_snle, infer_parameters_snle, generate_likelihood_training_data
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_utils import plot_training_history

# Parameter grid
NUM_SIMULATIONS = [1000000, 2000000]
WINDOW_SITES = [25, 50, 75,100]

# Test cases (all values ≤ 1)
TEST_CASES = [
    ("low_drift_high_bump", torch.tensor([0.2, 0.8, 0.3])),
    ("high_drift_low_bump", torch.tensor([0.8, 0.2, 0.3])),
    ("balanced", torch.tensor([0.5, 0.5, 0.3])),
]


def pre_generate_training_data(results_dir, logger):
    """
    Pre-generate all training data needed for the sweep.
    
    Generates max(NUM_SIMULATIONS) samples for each window_sites value.
    Saves data to results_dir for reuse.
    
    Returns:
        dict mapping window_sites -> (theta_samples, x_samples, x_mean, x_std)
    """
    logger.info("\n" + "="*80)
    logger.info("PRE-GENERATING TRAINING DATA")
    logger.info("="*80)
    
    simulator = PatchForagingDDM()
    prior = create_prior()
    
    max_sims = max(NUM_SIMULATIONS)
    training_data = {}
    
    for window_sites in WINDOW_SITES:
        logger.info(f"\nGenerating data for window_sites={window_sites} (n={max_sims})...")
        
        # Check if data already exists
        data_path = os.path.join(results_dir, f'training_data_sites{window_sites}.pkl')
        
        if os.path.exists(data_path):
            logger.info(f"  Loading existing data from {data_path}")
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            training_data[window_sites] = data
        else:
            # Generate new data
            theta_samples, x_samples, x_mean, x_std = generate_likelihood_training_data(
                simulator, prior, 
                num_simulations=max_sims,
                window_sites=window_sites,
                mode='multi'
            )
            
            data = {
                'theta_samples': theta_samples,
                'x_samples': x_samples,
                'x_mean': x_mean,
                'x_std': x_std,
                'num_simulations': max_sims,
                'window_sites': window_sites,
            }
            
            # Save data
            with open(data_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"  Saved to {data_path}")
            
            training_data[window_sites] = data
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING DATA GENERATION COMPLETE")
    logger.info("="*80)
    
    return training_data


def setup_logging(results_dir):
    """
    Setup logging to both console and file.
    """
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
    
    Computes Wasserstein distance and KL divergence for each summary statistic.
    """
    logger.info(f"  Computing generative metrics ({num_patches} patches)...")
    
    # Generate from true parameters
    true_patches = []
    for _ in range(num_patches):
        global_time = 0.0
        patch_start_time = 0.0
        _, _, stats = simulator._simulate_one_patch(true_theta, global_time, patch_start_time)
        true_patches.append(stats)
    true_data = torch.stack(true_patches).numpy()
    
    # Generate from posterior samples
    posterior_patches = []
    for i in range(num_patches):
        theta_sample = posterior_samples[i % len(posterior_samples)]
        global_time = 0.0
        patch_start_time = 0.0
        _, _, stats = simulator._simulate_one_patch(theta_sample, global_time, patch_start_time)
        posterior_patches.append(stats)
    posterior_data = torch.stack(posterior_patches).numpy()
    
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
    # 1. Generate observed data
    param_gen = simulator.walk_params(true_theta)
    _, observed_stats = simulator.simulate_trial(
        param_gen, 
        window_sites=100,  # Fixed for testing
        return_aggregate=True
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
    """
    Save individual result as pickle and append summary to CSV.
    """
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
        - exists: bool, whether model exists
        - model_dir: path to model directory
        - model_dict: loaded model dict if exists, None otherwise
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


def train_and_evaluate(num_sims, window_sites, results_dir, logger, training_data_dict, skip_if_exists=True):
    """
    Train SNLE model and evaluate on all test cases.
    
    Args:
        num_sims: Number of training simulations
        window_sites: Number of sites per simulation
        results_dir: Directory to save results
        logger: Logger instance
        training_data_dict: Pre-generated training data dict
        skip_if_exists: If True, skip training if model already exists
    
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
        
        # Check if we have training metrics, otherwise create dummy ones
        if 'training_metrics' in existing_model:
            training_metrics = existing_model['training_metrics']
        else:
            # Legacy model without training metrics saved
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
        from sbi.inference import SNLE
        simulator = PatchForagingDDM()
        prior = create_prior()
        
        # Create model-specific directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Get pre-generated training data and subset it
        logger.info(f"  Using pre-generated training data (subset: first {num_sims} samples)")
        full_data = training_data_dict[window_sites]
        
        theta_samples = full_data['theta_samples'][:num_sims]
        x_samples = full_data['x_samples'][:num_sims]
        
        # Use the normalization from the full dataset for consistency
        x_mean = full_data['x_mean']
        x_std = full_data['x_std']
        
        logger.info(f"  Subset shape: theta={theta_samples.shape}, x={x_samples.shape}")
        
        # Train model
        try:
            # Initialize SNLE
            logger.info(f"  Training SNLE (multi mode)...")
            inference = SNLE(prior=prior)
            
            # Add training data
            inference.append_simulations(theta_samples, x_samples)
            
            # Train with more detailed output
            likelihood_estimator = inference.train(
                training_batch_size=50,
                max_num_epochs=50,
                show_train_summary=True,
                stop_after_epochs=20,  # Early stopping patience
            )
            
            # Extract training history with correct keys (handle lists)
            best_val = inference._summary['best_validation_loss']
            if isinstance(best_val, list):
                best_val = best_val[0] if len(best_val) > 0 else None
            
            epochs_trained = inference._summary['epochs_trained']
            if isinstance(epochs_trained, list):
                epochs_trained = epochs_trained[0] if len(epochs_trained) > 0 else len(inference._summary['training_loss'])
            
            history = {
                'train_loss': inference._summary['training_loss'],
                'val_loss': inference._summary['validation_loss'],
                'epochs': list(range(len(inference._summary['training_loss']))),
                'best_val_loss': best_val,
                'epochs_trained': epochs_trained,
            }
            
            # Store training metrics
            training_metrics = {
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'best_val_loss': history['best_val_loss'],
                'epochs_trained': history['epochs_trained'],
            }
            
            logger.info(f"Training complete: val_loss={training_metrics['final_val_loss']:.4f}")
            
            # Save training history plot
            plot_path = os.path.join(model_dir, 'training_history.png')
            plot_training_history(history, mode='multi', save_path=plot_path)
            
            # Save the trained model
            model_save_path = os.path.join(model_dir, 'model.pkl')
            model_dict = {
                'inference': inference,
                'x_mean': x_mean,
                'x_std': x_std,
                'mode': 'multi',
                'num_simulations': num_sims,
                'window_sites': window_sites,
                'training_metrics': training_metrics,
            }
            torch.save(model_dict, model_save_path)
            logger.info(f"Model saved to: {model_save_path}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    # Setup simulator for evaluation
    simulator = PatchForagingDDM()
    
    # Evaluate on all test cases
    test_results = []
    for test_name, true_theta in TEST_CASES:
        logger.info(f"\nEvaluating on test case: {test_name}")
        
        try:
            metrics = evaluate_on_test_case(
                simulator, inference, x_mean, x_std,
                true_theta, test_name, logger
            )
            test_results.append(metrics)
            
        except Exception as e:
            logger.error(f"Evaluation failed for {test_name}: {e}")
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
    }
    
    # Save intermediate result
    save_intermediate_result(result, results_dir)
    
    return result


def create_summary_plots(results_dir, logger):
    """
    Create summary visualizations from sweep results.
    
    Generates:
    1. Heatmaps: MAE, RMSE, Coverage vs (num_sims, window_sites)
    2. Heatmaps: Wasserstein, KL divergence vs (num_sims, window_sites)
    3. Line plots: Metrics vs num_sims (for each window_sites)
    4. Line plots: Metrics vs window_sites (for each num_sims)
    """
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
        axes[idx].set_xticklabels(pivot.columns, rotation=45, ha='right')
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
    
    # 2. Line plots: Metrics vs num_sims (separate line for each window_sites)
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
    
    # 3. Line plots: Metrics vs window_sites (separate line for each num_sims)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        for num_sims in sorted(df_avg['num_simulations'].unique()):
            data = df_avg[df_avg['num_simulations'] == num_sims].sort_values('window_sites')
            axes[idx].plot(data['window_sites'], data[metric], 
                          marker='o', label=f'sims={num_sims}', linewidth=2)
        
        axes[idx].set_xlabel('window_sites', fontsize=10)
        axes[idx].set_ylabel(label, fontsize=10)
        axes[idx].set_title(f'{label} vs window_sites', fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    sites_path = os.path.join(results_dir, 'metrics_vs_window_sites.png')
    plt.savefig(sites_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: metrics_vs_window_sites.png")
    
    # 4. Per-test-case breakdown
    fig, axes = plt.subplots(len(TEST_CASES), 3, figsize=(18, 4*len(TEST_CASES)))
    if len(TEST_CASES) == 1:
        axes = axes.reshape(1, -1)
    
    for test_idx, (test_name, _) in enumerate(TEST_CASES):
        df_test = df[df['test_case'] == test_name].groupby(['num_simulations', 'window_sites']).agg({
            'mean_mae': 'mean',
            'mean_wasserstein': 'mean',
            'mean_kl': 'mean',
        }).reset_index()
        
        # MAE vs num_sims
        ax = axes[test_idx, 0]
        for window_sites in sorted(df_test['window_sites'].unique()):
            data = df_test[df_test['window_sites'] == window_sites].sort_values('num_simulations')
            ax.plot(data['num_simulations'], data['mean_mae'], marker='o', label=f'sites={window_sites}', linewidth=2)
        ax.set_xlabel('num_simulations')
        ax.set_ylabel('Mean MAE')
        ax.set_title(f'{test_name}: MAE vs num_simulations', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # Wasserstein vs num_sims
        ax = axes[test_idx, 1]
        for window_sites in sorted(df_test['window_sites'].unique()):
            data = df_test[df_test['window_sites'] == window_sites].sort_values('num_simulations')
            ax.plot(data['num_simulations'], data['mean_wasserstein'], marker='o', label=f'sites={window_sites}', linewidth=2)
        ax.set_xlabel('num_simulations')
        ax.set_ylabel('Mean Wasserstein')
        ax.set_title(f'{test_name}: Wasserstein vs num_simulations', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # KL vs num_sims
        ax = axes[test_idx, 2]
        for window_sites in sorted(df_test['window_sites'].unique()):
            data = df_test[df_test['window_sites'] == window_sites].sort_values('num_simulations')
            ax.plot(data['num_simulations'], data['mean_kl'], marker='o', label=f'sites={window_sites}', linewidth=2)
        ax.set_xlabel('num_simulations')
        ax.set_ylabel('Mean KL')
        ax.set_title(f'{test_name}: KL vs num_simulations', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    plt.tight_layout()
    testcase_path = os.path.join(results_dir, 'per_test_case_breakdown.png')
    plt.savefig(testcase_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: per_test_case_breakdown.png")
    
    logger.info("Summary visualizations complete!")


def run_parameter_sweep(base_dir='snle_parameter_sweep', skip_if_exists=True, resume_dir=None):
    """
    Run full parameter sweep experiment.
    Creates timestamped folder with all results, or resumes from existing directory.
    
    Args:
        base_dir: Base directory for saving results (only used if resume_dir is None)
        skip_if_exists: If True, skip training for models that already exist
        resume_dir: If provided, resume from this existing results directory
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
    logger.info("="*80)
    
    # Pre-generate all training data
    training_data_dict = pre_generate_training_data(results_dir, logger)
    
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
            
            # Check if already exists before calling train_and_evaluate
            exists, _, _ = check_existing_model(num_sims, window_sites, results_dir)
            if exists and skip_if_exists:
                num_skipped += 1
            
            result = train_and_evaluate(num_sims, window_sites, results_dir, logger, 
                                       training_data_dict, skip_if_exists=skip_if_exists)
            
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
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='SNLE Parameter Sweep Analysis')
    parser.add_argument('mode', nargs='?', default='full', choices=['test', 'force', 'full'],
                       help='Run mode: test (quick test), force (retrain all), full (default, skip existing)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from existing results directory (e.g., snle_parameter_sweep/sweep_20241108_143022)')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        # Quick test with reduced parameters
        print("Running quick test with reduced parameters...")
        NUM_SIMULATIONS = [1000, 5000]
        WINDOW_SITES = [10, 50]
        results_dir, all_results = run_parameter_sweep(
            base_dir='snle_parameter_sweep_test',
            resume_dir=args.resume
        )
    elif args.mode == 'force':
        # Force retraining even if models exist
        print("Running full sweep with forced retraining...")
        results_dir, all_results = run_parameter_sweep(
            skip_if_exists=False,
            resume_dir=args.resume
        )
    else:
        # Full parameter sweep (default: skip existing models)
        results_dir, all_results = run_parameter_sweep(
            resume_dir=args.resume
        )