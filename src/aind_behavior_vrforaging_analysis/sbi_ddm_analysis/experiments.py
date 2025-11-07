"""
Experiment scripts for patch foraging inference.

Simple, clear workflows for common tasks.
"""

import torch
from simulator import PatchForagingDDM, create_prior
from inference import train_sbi, infer_parameters, save_posterior, load_posterior
from pathlib import Path
from validation import run_sbc, print_correlations, plot_posterior, plot_pairplot


def experiment_basic_training(num_simulations: int = 50000):
    """
    Basic training experiment.
    
    Train SBI model and test on single window.
    """
    print("="*60)
    print(f"EXPERIMENT: Basic Training ({num_simulations} simulations)")
    print("="*60)
    
    # Setup
    simulator = PatchForagingDDM()
    prior = create_prior()
    
    # Train
    print("\n1. Training...")
    posterior_path = 'posterior_basic.pkl'

    if Path(posterior_path).exists():
        print("Loading existing posterior...")
        posterior = load_posterior(posterior_path)
    else:
        print("Training new posterior...")
        posterior = train_sbi(simulator, prior, num_simulations=num_simulations)
        save_posterior(posterior, posterior_path)
    
    # Test on single window
    print("\n2. Testing inference...")
    test_theta = torch.tensor([0.5, 0.6, 0.2])
    test_window = simulator.simulate_with_random_walk(test_theta, window_sites=300, random_walk_sigma=0.0)
    
    samples = infer_parameters(posterior, test_window, num_samples=2000)
    posterior_mean = samples.mean(dim=0)
    
    print(f"\nTrue theta:      {test_theta}")
    print(f"Posterior mean:  {posterior_mean}")
    print(f"Difference:      {(posterior_mean - test_theta).abs()}")
    
    # Analyze
    print("\n3. Checking correlations...")
    corr = print_correlations(samples)
    
    # Plot
    print("\n4. Generating plots...")
    plot_posterior(samples, true_theta=test_theta, save_path='posterior_basic.png')
    plot_pairplot(samples, true_theta=test_theta, save_path='pairplot_basic.png')
    
    print("\n" + "="*60)
    print("KEY RESULTS:")
    print(f"  drift_rate ↔ reward_bump correlation: r = {corr[0,1]:.3f}")
    print(f"  Goal: r < 0.5 (well-separated)")
    print("="*60)
    
    return posterior


def experiment_validation(posterior, num_tests: int = 50):
    """
    Validation experiment using SBC.
    
    Tests if posterior is well-calibrated.
    """
    print("\n" + "="*60)
    print(f"EXPERIMENT: Validation ({num_tests} SBC tests)")
    print("="*60)
    
    simulator = PatchForagingDDM()
    prior = create_prior()
    
    ranks = run_sbc(simulator, prior, posterior, num_tests=num_tests)
    
    print("\n✓ Validation complete - check if ranks are ~uniform")
    
    return ranks


def experiment_compare_parameters():
    """
    Compare inference quality across different parameter regimes.
    
    Tests if some parameters are harder to infer than others.
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Parameter Regime Comparison")
    print("="*60)
    
    simulator = PatchForagingDDM()
    prior = create_prior()
    
    # Train once
    print("\n1. Training model...")
    posterior_path = 'posterior_basic.pkl'

    if Path(posterior_path).exists():
        print("Loading existing posterior...")
        posterior = load_posterior(posterior_path)
    else:
        print("Training new posterior...")
        posterior = train_sbi(simulator, prior, num_simulations=num_simulations)
        save_posterior(posterior, posterior_path)
    
    # Test on different parameter combinations
    test_cases = [
        ("Low drift, high reward_bump", torch.tensor([0.2, 1.2, 0.3])),
        ("High drift, low reward_bump", torch.tensor([1.2, 0.2, 0.3])),
        ("Balanced", torch.tensor([0.6, 0.6, 0.3])),
    ]
    
    print("\n2. Testing different parameter regimes...")
    for name, theta in test_cases:
        print(f"\n--- {name} ---")
        window = simulator.simulate_with_random_walk(theta, window_sites=300, random_walk_sigma=0.0)
        samples = infer_parameters(posterior, window, num_samples=1000)
        mean_estimate = samples.mean(dim=0)
        
        print(f"True:     {theta}")
        print(f"Estimate: {mean_estimate}")
        print(f"Error:    {(mean_estimate - theta).abs()}")
    
    print("\n✓ Comparison complete")


def experiment_full_pipeline(num_simulations: int = 50000, num_sbc_tests: int = 50):
    """
    Full pipeline: train, validate, analyze.
    
    This is the main experiment to run.
    """
    print("="*60)
    print("FULL PIPELINE EXPERIMENT")
    print("="*60)
    
    # Train
    posterior = experiment_basic_training(num_simulations=num_simulations)
    
    # Validate
    experiment_validation(posterior, num_tests=num_sbc_tests)
    
    # Compare regimes
    experiment_compare_parameters()
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("Generated files:")
    print("  - posterior_basic.png")
    print("  - pairplot_basic.png")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'quick':
            # Quick test with small dataset
            print("Running quick test (5K simulations)...")
            experiment_basic_training(num_simulations=5000)
        elif sys.argv[1] == 'full':
            # Full pipeline
            experiment_full_pipeline(num_simulations=50000, num_sbc_tests=50)
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python experiments.py [quick|full]")
    else:
        # Default: basic training with 50K
        experiment_basic_training(num_simulations=50000)