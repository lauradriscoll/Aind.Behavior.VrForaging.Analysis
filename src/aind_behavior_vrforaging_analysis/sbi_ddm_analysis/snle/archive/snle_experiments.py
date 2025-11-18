"""
Experiment scripts for SNLE patch foraging inference.

Purpose: Pre-defined workflows for common SNLE experiments.
"""

import torch
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM, create_prior
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.archive.snle_inference import train_snle, infer_parameters_snle, load_snle_model, save_snle_model
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_utils  import (plot_training_history, plot_posterior_pairplot, 
                        compare_snle_vs_simulator, print_inference_summary,
                        plot_posterior_distributions)


def experiment_single_patch_training(num_simulations=20000, window_sites=100):
    """
    Train SNLE for single-patch inference.
    
    Args:
        num_simulations: Number of training samples
        window_sites: Number of sites per simulation
    
    Returns:
        inference: Trained SNLE model
        x_mean, x_std: Normalization parameters
        training_history: Training diagnostics
    """
    print("="*60)
    print(f"EXPERIMENT: Single-Patch SNLE Training ({num_simulations} simulations)")
    print("="*60)
    
    # Setup
    simulator = PatchForagingDDM()
    prior = create_prior()
    
    # Train
    print("\n1. Training SNLE (single-patch mode)...")
    likelihood_estimator, inference, x_mean, x_std, history = train_snle(
        simulator, prior, 
        num_simulations=num_simulations,
        window_sites=window_sites,
        mode='single'
    )
    
    # Plot training history
    print("\n2. Plotting training history...")
    model_dir = save_snle_model(inference, x_mean, x_std, mode='single')
    _, _, _, _, analysis_dir = load_snle_model(model_dir)
    plot_training_history(history, mode='single', analysis_dir=analysis_dir)
    
    print("\n" + "="*60)
    print("Single-patch training complete!")
    print(f"Model directory: {model_dir}")
    print(f"Analysis directory: {analysis_dir}")
    print("="*60)
    
    return inference, x_mean, x_std, history, model_dir


def experiment_multi_patch_training(num_simulations=20000, window_sites=100):
    """
    Train SNLE for multi-patch inference.
    
    Args:
        num_simulations: Number of training samples
        window_sites: Number of sites per simulation
    
    Returns:
        inference: Trained SNLE model
        x_mean, x_std: Normalization parameters
        training_history: Training diagnostics
    """
    print("="*60)
    print(f"EXPERIMENT: Multi-Patch SNLE Training ({num_simulations} simulations)")
    print("="*60)
    
    # Setup
    simulator = PatchForagingDDM()
    prior = create_prior()
    
    # Train
    print("\n1. Training SNLE (multi-patch mode)...")
    likelihood_estimator, inference, x_mean, x_std, history = train_snle(
        simulator, prior, 
        num_simulations=num_simulations,
        window_sites=window_sites,
        mode='multi'
    )
    
    # Plot training history
    print("\n2. Plotting training history...")
    model_dir = save_snle_model(inference, x_mean, x_std, mode='multi')
    _, _, _, _, analysis_dir = load_snle_model(model_dir)
    plot_training_history(history, mode='multi', analysis_dir=analysis_dir)
    
    print("\n" + "="*60)
    print("Multi-patch training complete!")
    print(f"Model directory: {model_dir}")
    print(f"Analysis directory: {analysis_dir}")
    print("="*60)
    
    return inference, x_mean, x_std, history, model_dir


def experiment_test_inference(inference, x_mean, x_std, mode='single', 
                              analysis_dir=None, test_thetas=None):
    """
    Test SNLE inference on known parameter values.
    
    Args:
        inference: Trained SNLE model
        x_mean, x_std: Normalization parameters
        mode: 'single' or 'multi'
        analysis_dir: Directory to save analysis plots (if None, uses current directory)
        test_thetas: List of test parameter values (optional)
    
    Returns:
        results: List of inference results
    """
    import os
    
    print("\n" + "="*60)
    print(f"EXPERIMENT: Testing Inference ({mode} mode)")
    print("="*60)
    
    # Setup
    simulator = PatchForagingDDM()
    prior = create_prior()
    
    # Default test cases
    if test_thetas is None:
        test_thetas = [
            ("Low_drift_high_reward_bump", torch.tensor([0.2, 1.2, 0.3])),
            ("High_drift_low_reward_bump", torch.tensor([1.2, 0.2, 0.3])),
            ("Balanced", torch.tensor([0.6, 0.6, 0.3])),
        ]
    
    results = []
    
    for test_name, true_theta in test_thetas:
        print(f"\n{'='*60}")
        print(f"Test Case: {test_name}")
        print(f"True theta: {true_theta}")
        print(f"{'='*60}")
        
        # Generate observed data
        print("\n1. Generating observed data...")
        param_gen = simulator.walk_params_params(true_theta)
        _, observed_stats = simulator.simulate_trial(
            param_gen, 
            window_sites=100,
            return_aggregate=(mode == 'multi')
        )
        print(f"   Observed stats: {observed_stats}")
        
        # Infer parameters
        print("\n2. Running MCMC inference...")
        posterior_samples = infer_parameters_snle(
            inference, observed_stats,
            x_mean, x_std,
            num_samples=1000, warmup_steps=200
        )
        
        # Analyze results
        print("\n3. Analyzing results...")
        print_inference_summary(posterior_samples, true_theta)
        
        # Determine save paths
        if analysis_dir:
            posterior_path = os.path.join(analysis_dir, f'{test_name}_posterior.png')
            pairplot_path = os.path.join(analysis_dir, f'{test_name}_pairplot.png')
            comparison_path = os.path.join(analysis_dir, f'{test_name}_comparison.png')
        else:
            posterior_path = f'{test_name}_posterior.png'
            pairplot_path = f'{test_name}_pairplot.png'
            comparison_path = f'{test_name}_comparison.png'
        
        # Plot posterior
        print("\n4. Generating plots...")
        plot_posterior_distributions(
            posterior_samples, true_theta,
            save_path=posterior_path
        )
        
        plot_posterior_pairplot(
            prior, posterior_samples, true_theta,
            save_path=pairplot_path
        )
        
        # Compare SNLE vs simulator
        print("\n5. Comparing SNLE vs simulator...")
        compare_snle_vs_simulator(
            simulator, true_theta, posterior_samples,
            num_patches=100,
            save_path=comparison_path
        )
        
        results.append({
            'name': test_name,
            'true_theta': true_theta,
            'posterior_samples': posterior_samples,
            'posterior_mean': posterior_samples.mean(dim=0),
            'posterior_std': posterior_samples.std(dim=0),
        })
    
    print("\n" + "="*60)
    print("Testing complete!")
    if analysis_dir:
        print(f"All plots saved to: {analysis_dir}")
    print("="*60)
    
    return results


def experiment_compare_single_vs_multi(num_simulations = 100000):
    """
    Compare single-patch vs multi-patch SNLE performance.
    
    Trains both models and tests them on the same data.
    """
    print("="*60)
    print("EXPERIMENT: Single-Patch vs Multi-Patch Comparison")
    print("="*60)
    
    # Setup
    simulator = PatchForagingDDM()
    prior = create_prior()
    
    # Train both models
    print("\n1. Training single-patch model...")
    _, inference_single, x_mean_single, x_std_single, _ = train_snle(
        simulator, prior, num_simulations=num_simulations, mode='single'
    )
    
    print("\n2. Training multi-patch model...")
    _, inference_multi, x_mean_multi, x_std_multi, _ = train_snle(
        simulator, prior, num_simulations=num_simulations, mode='multi'
    )
    
    # Test on same parameters
    true_theta = torch.tensor([0.6, 0.8, 0.3])
    print(f"\n3. Testing both models on same data...")
    print(f"   True theta: {true_theta}")
    
    # Single-patch inference
    param_gen = simulator.walk_params(true_theta)
    _, obs_single = simulator.simulate_trial(param_gen, 100, return_aggregate=False)
    samples_single = infer_parameters_snle(
        inference_single, obs_single, x_mean_single, x_std_single,
        num_samples=1000, warmup_steps=200
    )
    
    # Multi-patch inference
    param_gen = simulator.walk_params(true_theta)
    _, obs_multi = simulator.simulate_trial(param_gen, 100, return_aggregate=True)
    samples_multi = infer_parameters_snle(
        inference_multi, obs_multi, x_mean_multi, x_std_multi,
        num_samples=1000, warmup_steps=200
    )
    
    # Compare results
    print("\n4. Comparison results:")
    print("\nSingle-Patch Model:")
    print_inference_summary(samples_single, true_theta)
    
    print("\nMulti-Patch Model:")
    print_inference_summary(samples_multi, true_theta)
    
    # Calculate which is better
    error_single = (samples_single.mean(dim=0) - true_theta).abs().mean()
    error_multi = (samples_multi.mean(dim=0) - true_theta).abs().mean()
    
    print(f"\n{'='*60}")
    print("WINNER:")
    if error_single < error_multi:
        print("✓ Single-patch model has lower error")
        print(f"  Single: {error_single:.4f}, Multi: {error_multi:.4f}")
    else:
        print("✓ Multi-patch model has lower error")
        print(f"  Single: {error_single:.4f}, Multi: {error_multi:.4f}")
    print(f"{'='*60}")


def experiment_full_pipeline(num_simulations=20000, mode='multi'):
    """
    Complete SNLE pipeline: train, test, and analyze.
    
    Args:
        num_simulations: Number of training samples
        mode: 'single' or 'multi'
    """
    print("="*60)
    print(f"FULL SNLE PIPELINE ({mode} mode)")
    print("="*60)
    
    # Train
    if mode == 'single':
        inference, x_mean, x_std, history, model_dir = experiment_single_patch_training(num_simulations)
    else:
        inference, x_mean, x_std, history, model_dir = experiment_multi_patch_training(num_simulations)
    
    # Get analysis directory
    _, _, _, _, analysis_dir = load_snle_model(model_dir)
    
    # Test
    results = experiment_test_inference(inference, x_mean, x_std, mode=mode, analysis_dir=analysis_dir)
    
    print("\n" + "="*60)
    print("FULL PIPELINE COMPLETE!")
    print(f"Trained and tested {mode}-patch SNLE model")
    print(f"Model directory: {model_dir}")
    print(f"Analysis directory: {analysis_dir}")
    print(f"Generated plots for {len(results)} test cases")
    print("="*60)
    
    return inference, x_mean, x_std, results, model_dir


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'single':
            # Train single-patch model
            experiment_single_patch_training(num_simulations=100000)
        elif sys.argv[1] == 'multi':
            # Train multi-patch model
            experiment_multi_patch_training(num_simulations=100000)
        elif sys.argv[1] == 'compare':
            # Compare both models
            experiment_compare_single_vs_multi(num_simulations=100000)
        elif sys.argv[1] == 'full':
            # Full pipeline
            mode = sys.argv[2] if len(sys.argv) > 2 else 'multi'
            experiment_full_pipeline(num_simulations=100000, mode=mode)
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python snle_experiments.py [single|multi|compare|full]")
    else:
        # Default: run quick test
        print("Running quick test (small dataset)...")
        experiment_full_pipeline(num_simulations=100000, mode='single')