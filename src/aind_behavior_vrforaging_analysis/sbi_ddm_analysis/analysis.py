"""
Analysis script to evaluate how training sample size and test size impact MNLE model accuracy.

This script provides two main operations:
1. Training: Train multiple MNLE models with different training sample sizes
2. Analysis: Evaluate all trained models on test datasets of varying sizes

Usage:
    # Train all models
    python analysis.py --mode train --output_dir analysis_results/
    
    # Analyze existing models
    python analysis.py --mode analyze --input_dir analysis_results/
    
    # Do both
    python analysis.py --mode both --output_dir analysis_results/
"""

import sys
sys.path.append('../src')

import argparse
import os
import pickle
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

from sbi.inference import MNLE
from sbi.inference.posteriors import MCMCPosteriorParameters

# Import from training script
from train_mnle import (
    setup_ddm,
    generate_training_data,
    train_mnle_model
)

# Import simulator components
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import DDMSimulator, create_ddm_prior


# ============================================================================
# Configuration
# ============================================================================

TRAINING_SAMPLE_SIZES = [1000, 5000, 10000, 25000, 50000]
NUM_RUNS_PER_SIZE = 3
TEST_TRIAL_SIZES = [10, 20, 30, 40]
MAX_REWARDS = 25  # Hard-coded based on prior analysis
NUM_POSTERIOR_SAMPLES = 1000
PARAM_NAMES = ['drift', 'reward_pulse']


# ============================================================================
# Phase 1: Model Training
# ============================================================================

def train_all_models(output_dir, sample_sizes=TRAINING_SAMPLE_SIZES, 
                     num_runs=NUM_RUNS_PER_SIZE, base_seed=0):
    """
    Train multiple MNLE models with different training sample sizes.
    
    Args:
        output_dir: Base directory for saving models
        sample_sizes: List of training sample sizes to test
        num_runs: Number of independent runs per sample size
        base_seed: Base random seed (will be incremented for each run)
    
    Returns:
        training_log: List of dicts with training info for each model
    """
    print("="*70)
    print("PHASE 1: TRAINING MODELS")
    print("="*70)
    print(f"Sample sizes: {sample_sizes}")
    print(f"Runs per size: {num_runs}")
    print(f"Total models to train: {len(sample_sizes) * num_runs}")
    print()
    
    # Create models directory
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Setup simulator and prior once
    simulator, prior = setup_ddm()
    
    training_log = []
    
    # Train models
    for n_samples in sample_sizes:
        for run_idx in range(num_runs):
            seed = base_seed + run_idx
            
            print(f"\n{'='*70}")
            print(f"Training: n_samples={n_samples}, run={run_idx+1}/{num_runs}, seed={seed}")
            print(f"{'='*70}")
            
            # Create output directory for this model
            model_name = f"n{n_samples}_run{run_idx}"
            model_dir = os.path.join(models_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Generate training data
            start_time = time.time()
            theta, x, clamping_info = generate_training_data(simulator, prior, n_samples, MAX_REWARDS, seed=seed)
            
            # Train model
            estimator, trainer = train_mnle_model(theta, x, MAX_REWARDS)
            training_time = time.time() - start_time
            
            # Save model
            model_path = os.path.join(model_dir, 'mnle_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'estimator': estimator,
                    'trainer': trainer,
                    'prior': prior,
                }, f)
            
            # Save metadata
            metadata = {
                'n_samples': n_samples,
                'run_idx': run_idx,
                'seed': seed,
                'max_rewards': MAX_REWARDS,
                'training_time': training_time,
                'clamping_info': clamping_info,
                'timestamp': datetime.now().isoformat()
            }

            metadata_path = os.path.join(model_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\nModel saved to: {model_dir}")
            print(f"Training time: {training_time:.2f} seconds")
            
            # Add to log
            training_log.append({
                'model_name': model_name,
                'model_dir': model_dir,
                'n_samples': n_samples,
                'run_idx': run_idx,
                'seed': seed,
                'training_time': training_time
            })
    
    # Save training log
    log_path = os.path.join(output_dir, 'training_log.pkl')
    with open(log_path, 'wb') as f:
        pickle.dump(training_log, f)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Trained {len(training_log)} models")
    print(f"Models saved to: {models_dir}")
    
    return training_log


# ============================================================================
# Phase 2: Model Analysis
# ============================================================================

def generate_test_datasets(simulator, prior, test_sizes=TEST_TRIAL_SIZES, 
                          num_test_params=5, base_seed=1000):
    """
    Generate multiple test datasets with different numbers of trials.
    
    Args:
        simulator: DDMSimulator instance
        prior: Prior distribution
        test_sizes: List of test trial sizes
        num_test_params: Number of different parameter sets to test
        base_seed: Base random seed for test data generation
    
    Returns:
        test_datasets: List of dicts with test data info
    """
    print("\nGenerating test datasets...")
    print(f"Test sizes: {test_sizes}")
    print(f"Number of parameter sets: {num_test_params}")
    
    test_datasets = []
    
    for param_idx in range(num_test_params):
        seed = base_seed + param_idx
        torch.manual_seed(seed)
        
        # Sample true parameters
        theta_true = prior.sample((1,))
        
        # Generate observations for each test size
        for n_trials in test_sizes:
            torch.manual_seed(seed)  # Reset seed for reproducibility
            x_observed = torch.stack([simulator(theta_true[0]) for _ in range(n_trials)])
            
            test_datasets.append({
                'param_idx': param_idx,
                'n_trials': n_trials,
                'seed': seed,
                'theta_true': theta_true.numpy(),
                'x_observed': x_observed.numpy()
            })
    
    print(f"Generated {len(test_datasets)} test datasets")
    return test_datasets


def evaluate_single_model(model_info, test_dataset, num_posterior_samples=NUM_POSTERIOR_SAMPLES):
    """
    Evaluate a single model on a single test dataset.
    
    Args:
        model_info: Dict with model path and metadata
        test_dataset: Dict with test data
        num_posterior_samples: Number of posterior samples to draw
    
    Returns:
        results: Dict with evaluation metrics
    """
    # Load model
    model_path = os.path.join(model_info['model_dir'], 'mnle_model.pkl')
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    trainer = model_data['trainer']
    prior = model_data['prior']
    
    # Convert test data to tensors
    theta_true = torch.tensor(test_dataset['theta_true'], dtype=torch.float32)
    x_observed = torch.tensor(test_dataset['x_observed'], dtype=torch.float32)
    
    # Configure MCMC
    params = MCMCPosteriorParameters(
        method="slice_np_vectorized",
        num_chains=5,
        thin=10,
        warmup_steps=100,
        init_strategy="proposal"
    )
    
    # Build posterior
    posterior = trainer.build_posterior(prior=prior, posterior_parameters=params)
    
    # Sample from posterior
    try:
        samples = posterior.sample(
            (num_posterior_samples,), 
            x=x_observed, 
            show_progress_bars=False
        )
        samples_np = samples.numpy()
        
        # Calculate metrics
        posterior_mean = samples_np.mean(axis=0)
        posterior_std = samples_np.std(axis=0)
        true_params = theta_true[0].numpy()
        
        # Absolute error
        abs_error = np.abs(posterior_mean - true_params)
        
        # Relative error (%)
        rel_error = 100 * abs_error / (np.abs(true_params) + 1e-8)
        
        # Z-score
        z_score = abs_error / (posterior_std + 1e-8)
        
        # Coverage (95% credible interval)
        lower_bound = np.percentile(samples_np, 2.5, axis=0)
        upper_bound = np.percentile(samples_np, 97.5, axis=0)
        coverage = (true_params >= lower_bound) & (true_params <= upper_bound)
        
        results = {
            'success': True,
            'posterior_mean': posterior_mean,
            'posterior_std': posterior_std,
            'true_params': true_params,
            'abs_error': abs_error,
            'rel_error': rel_error,
            'z_score': z_score,
            'coverage': coverage,
            'samples': samples_np,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
    except Exception as e:
        print(f"Error during inference: {e}")
        results = {
            'success': False,
            'error': str(e)
        }
    
    return results


def analyze_all_models(input_dir, test_datasets=None, num_posterior_samples=NUM_POSTERIOR_SAMPLES):
    """
    Analyze all trained models on all test datasets.
    
    Args:
        input_dir: Directory containing trained models
        test_datasets: List of test datasets (will generate if None)
        num_posterior_samples: Number of posterior samples per evaluation
    
    Returns:
        all_results: List of dicts with all evaluation results
    """
    print("\n" + "="*70)
    print("PHASE 2: ANALYZING MODELS")
    print("="*70)
    
    # Load training log
    log_path = os.path.join(input_dir, 'training_log.pkl')
    if os.path.exists(log_path):
        with open(log_path, 'rb') as f:
            training_log = pickle.load(f)
        print(f"Loaded {len(training_log)} trained models")
    else:
        print("Error: training_log.pkl not found. Run training first.")
        return None
    
    # Generate test datasets if not provided
    if test_datasets is None:
        simulator, prior = setup_ddm()
        test_datasets = generate_test_datasets(simulator, prior)
    
    # Create results directory
    results_dir = os.path.join(input_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate all model-test combinations
    all_results = []
    total_evals = len(training_log) * len(test_datasets)
    
    print(f"\nEvaluating {len(training_log)} models on {len(test_datasets)} test datasets")
    print(f"Total evaluations: {total_evals}")
    print()
    
    with tqdm(total=total_evals, desc="Evaluating") as pbar:
        for model_info in training_log:
            for test_dataset in test_datasets:
                # Evaluate
                eval_results = evaluate_single_model(
                    model_info, test_dataset, num_posterior_samples
                )
                
                # Combine info
                result = {
                    'model_name': model_info['model_name'],
                    'n_training': model_info['n_samples'],
                    'run_idx': model_info['run_idx'],
                    'training_time': model_info['training_time'],
                    'n_test_trials': test_dataset['n_trials'],
                    'param_idx': test_dataset['param_idx'],
                    'test_seed': test_dataset['seed'],
                }
                result.update(eval_results)
                
                all_results.append(result)
                pbar.update(1)
    
    # Save all results
    results_path = os.path.join(results_dir, 'all_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved all results to: {results_path}")
    
    return all_results


def create_summary_dataframe(all_results):
    """
    Create a summary DataFrame from all results.
    
    Args:
        all_results: List of result dicts
    
    Returns:
        df: pandas DataFrame with summary statistics
    """
    rows = []
    
    for result in all_results:
        if not result['success']:
            continue
        
        for param_idx, param_name in enumerate(PARAM_NAMES):
            row = {
                'model_name': result['model_name'],
                'n_training': result['n_training'],
                'run_idx': result['run_idx'],
                'n_test_trials': result['n_test_trials'],
                'param_idx': result['param_idx'],
                'parameter': param_name,
                'true_value': result['true_params'][param_idx],
                'estimated_value': result['posterior_mean'][param_idx],
                'posterior_std': result['posterior_std'][param_idx],
                'abs_error': result['abs_error'][param_idx],
                'rel_error': result['rel_error'][param_idx],
                'z_score': result['z_score'][param_idx],
                'coverage': result['coverage'][param_idx],
                'training_time': result['training_time']
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


# ============================================================================
# Visualization
# ============================================================================

def plot_error_vs_training_size(df, output_dir):
    """Plot parameter estimation error vs. training sample size."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for param_idx, param_name in enumerate(PARAM_NAMES):
        ax = axes[param_idx]
        param_df = df[df['parameter'] == param_name]
        
        # Plot for each test size
        for n_test in sorted(param_df['n_test_trials'].unique()):
            test_df = param_df[param_df['n_test_trials'] == n_test]
            
            # Group by training size and compute mean/std
            grouped = test_df.groupby('n_training')['abs_error'].agg(['mean', 'std', 'count'])
            sem = grouped['std'] / np.sqrt(grouped['count'])
            
            ax.errorbar(grouped.index, grouped['mean'], yerr=sem, 
                       marker='o', label=f'{n_test} test trials', capsize=5)
        
        ax.set_xlabel('Number of Training Samples')
        ax.set_ylabel('Absolute Parameter Error')
        ax.set_title(f'{param_name.capitalize()} Parameter Recovery')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'error_vs_training_size.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def plot_error_vs_test_size(df, output_dir):
    """Plot parameter estimation error vs. test trial size."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for param_idx, param_name in enumerate(PARAM_NAMES):
        ax = axes[param_idx]
        param_df = df[df['parameter'] == param_name]
        
        # Plot for each training size
        for n_train in sorted(param_df['n_training'].unique()):
            train_df = param_df[param_df['n_training'] == n_train]
            
            # Group by test size and compute mean/std
            grouped = train_df.groupby('n_test_trials')['abs_error'].agg(['mean', 'std', 'count'])
            sem = grouped['std'] / np.sqrt(grouped['count'])
            
            ax.errorbar(grouped.index, grouped['mean'], yerr=sem,
                       marker='o', label=f'{n_train} train samples', capsize=5)
        
        ax.set_xlabel('Number of Test Trials')
        ax.set_ylabel('Absolute Parameter Error')
        ax.set_title(f'{param_name.capitalize()} Parameter Recovery')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'error_vs_test_size.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def plot_uncertainty_analysis(df, output_dir):
    """Plot posterior uncertainty vs. training and test sizes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for param_idx, param_name in enumerate(PARAM_NAMES):
        # Plot 1: Uncertainty vs training size
        ax = axes[param_idx, 0]
        param_df = df[df['parameter'] == param_name]
        
        for n_test in sorted(param_df['n_test_trials'].unique()):
            test_df = param_df[param_df['n_test_trials'] == n_test]
            grouped = test_df.groupby('n_training')['posterior_std'].agg(['mean', 'std', 'count'])
            sem = grouped['std'] / np.sqrt(grouped['count'])
            
            ax.errorbar(grouped.index, grouped['mean'], yerr=sem,
                       marker='o', label=f'{n_test} test trials', capsize=5)
        
        ax.set_xlabel('Number of Training Samples')
        ax.set_ylabel('Posterior Std Dev')
        ax.set_title(f'{param_name.capitalize()}: Uncertainty vs Training Size')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Uncertainty vs test size
        ax = axes[param_idx, 1]
        
        for n_train in sorted(param_df['n_training'].unique()):
            train_df = param_df[param_df['n_training'] == n_train]
            grouped = train_df.groupby('n_test_trials')['posterior_std'].agg(['mean', 'std', 'count'])
            sem = grouped['std'] / np.sqrt(grouped['count'])
            
            ax.errorbar(grouped.index, grouped['mean'], yerr=sem,
                       marker='o', label=f'{n_train} train samples', capsize=5)
        
        ax.set_xlabel('Number of Test Trials')
        ax.set_ylabel('Posterior Std Dev')
        ax.set_title(f'{param_name.capitalize()}: Uncertainty vs Test Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'uncertainty_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def plot_coverage_heatmap(df, output_dir):
    """Plot coverage probability as heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for param_idx, param_name in enumerate(PARAM_NAMES):
        ax = axes[param_idx]
        param_df = df[df['parameter'] == param_name]
        
        # Create pivot table for heatmap
        coverage_pivot = param_df.pivot_table(
            values='coverage',
            index='n_test_trials',
            columns='n_training',
            aggfunc='mean'
        )
        
        sns.heatmap(coverage_pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                   vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Coverage Probability'})
        ax.set_xlabel('Number of Training Samples')
        ax.set_ylabel('Number of Test Trials')
        ax.set_title(f'{param_name.capitalize()}: 95% CI Coverage')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'coverage_heatmap.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def plot_z_score_analysis(df, output_dir):
    """Plot z-score distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for param_idx, param_name in enumerate(PARAM_NAMES):
        ax = axes[param_idx]
        param_df = df[df['parameter'] == param_name]
        
        # Box plot of z-scores by training size
        training_sizes = sorted(param_df['n_training'].unique())
        z_scores_by_size = [
            param_df[param_df['n_training'] == n]['z_score'].values
            for n in training_sizes
        ]
        
        ax.boxplot(z_scores_by_size, labels=training_sizes)
        ax.axhline(y=1.96, color='r', linestyle='--', label='95% threshold')
        ax.axhline(y=-1.96, color='r', linestyle='--')
        ax.set_xlabel('Number of Training Samples')
        ax.set_ylabel('Z-Score')
        ax.set_title(f'{param_name.capitalize()}: Z-Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'z_score_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def plot_training_time_analysis(df, output_dir):
    """Plot training time vs sample size."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Get unique training times per model
    time_df = df.drop_duplicates(subset=['model_name'])[['n_training', 'training_time']]
    
    grouped = time_df.groupby('n_training')['training_time'].agg(['mean', 'std', 'count'])
    sem = grouped['std'] / np.sqrt(grouped['count'])
    
    ax.errorbar(grouped.index, grouped['mean'], yerr=sem,
               marker='o', capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time vs Sample Size')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_time_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def create_all_plots(all_results, output_dir):
    """Create all analysis plots."""
    print("\n" + "="*70)
    print("CREATING PLOTS")
    print("="*70)
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create summary DataFrame
    df = create_summary_dataframe(all_results)
    
    # Save summary CSV
    csv_path = os.path.join(output_dir, 'metrics_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV to: {csv_path}")
    
    # Create plots
    print("\nGenerating plots...")
    plot_error_vs_training_size(df, plots_dir)
    plot_error_vs_test_size(df, plots_dir)
    plot_uncertainty_analysis(df, plots_dir)
    plot_coverage_heatmap(df, plots_dir)
    plot_z_score_analysis(df, plots_dir)
    plot_training_time_analysis(df, plots_dir)
    
    print(f"\nAll plots saved to: {plots_dir}")
    
    return df


def print_summary_statistics(df):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for param_name in PARAM_NAMES:
        param_df = df[df['parameter'] == param_name]
        
        print(f"\n{param_name.upper()}:")
        print("-" * 70)
        
        print("\nMean Absolute Error by Training Size:")
        error_by_train = param_df.groupby('n_training')['abs_error'].mean()
        for n_train, error in error_by_train.items():
            print(f"  {n_train:6d} samples: {error:.4f}")
        
        print("\nMean Absolute Error by Test Size:")
        error_by_test = param_df.groupby('n_test_trials')['abs_error'].mean()
        for n_test, error in error_by_test.items():
            print(f"  {n_test:2d} trials: {error:.4f}")
        
        print("\nCoverage Rate by Training Size:")
        coverage_by_train = param_df.groupby('n_training')['coverage'].mean()
        for n_train, coverage in coverage_by_train.items():
            print(f"  {n_train:6d} samples: {coverage:.2%}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze how training and test sample sizes impact MNLE accuracy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models
  python analysis.py --mode train --output_dir analysis_results/
  
  # Analyze existing models
  python analysis.py --mode analyze --input_dir analysis_results/
  
  # Do both
  python analysis.py --mode both --output_dir analysis_results/
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'analyze', 'both'],
                       help='Operation mode: train models, analyze models, or both')
    parser.add_argument('--output_dir', type=str, default='analysis_results/',
                       help='Output directory (for train/both modes)')
    parser.add_argument('--input_dir', type=str, default='analysis_results/',
                       help='Input directory with trained models (for analyze mode)')
    parser.add_argument('--base_seed', type=int, default=0,
                       help='Base random seed for training')
    parser.add_argument('--test_seed', type=int, default=1000,
                       help='Base random seed for test data generation')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.mode in ['train', 'both']:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Execute based on mode
    if args.mode == 'train':
        print("Mode: TRAINING ONLY")
        train_all_models(args.output_dir, base_seed=args.base_seed)
        print("\nTraining complete! Run with --mode analyze to evaluate models.")
    
    elif args.mode == 'analyze':
        print("Mode: ANALYSIS ONLY")
        all_results = analyze_all_models(args.input_dir)
        if all_results:
            df = create_all_plots(all_results, args.input_dir)
            print_summary_statistics(df)
    
    elif args.mode == 'both':
        print("Mode: TRAINING + ANALYSIS")
        
        # Phase 1: Training
        train_all_models(args.output_dir, base_seed=args.base_seed)
        
        # Phase 2: Analysis
        all_results = analyze_all_models(args.output_dir)
        if all_results:
            df = create_all_plots(all_results, args.output_dir)
            print_summary_statistics(df)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()