"""
Benchmark JAX vs PyTorch simulator performance.

Usage:
    python benchmark_simulators.py
"""

import time
import torch
import numpy as np


def benchmark_pytorch():
    """Benchmark PyTorch simulator"""
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM, create_prior
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_inference import generate_likelihood_training_data
    
    print("\n" + "="*80)
    print("PYTORCH SIMULATOR")
    print("="*80)
    
    simulator = PatchForagingDDM()
    prior = create_prior()
    
    # Warmup
    print("\nWarming up...")
    _, _, _, _ = generate_likelihood_training_data(simulator, prior, num_simulations=100, window_sites=100, mode='multi')
    
    # Benchmark different sizes
    for num_sims in [1000, 10000, 50000]:
        print(f"\nGenerating {num_sims} simulations...")
        start = time.time()
        theta, x, x_mean, x_std = generate_likelihood_training_data(
            simulator, prior, 
            num_simulations=num_sims, 
            window_sites=100, 
            mode='multi'
        )
        elapsed = time.time() - start
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Rate: {num_sims/elapsed:.1f} sims/sec")
        print(f"  Shape: theta={theta.shape}, x={x.shape}")


def benchmark_jax():
    """Benchmark JAX simulator"""
    try:
        from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator_jax import PatchForagingDDMJax, create_prior_jax
        import jax.random as jax_random
        import jax
        
        print("\n" + "="*80)
        print("JAX SIMULATOR")
        print("="*80)
        print(f"JAX devices: {jax.devices()}")
        
        simulator = PatchForagingDDMJax()
        prior_low, prior_high = create_prior_jax()
        key = jax_random.PRNGKey(42)
        
        # Warmup (important for JAX JIT compilation)
        print("\nWarming up (JIT compiling)...")
        key, subkey = jax_random.split(key)
        _, _, _, _ = simulator.generate_training_data(
            subkey, prior_low, prior_high,
            num_simulations=100,
            window_sites=100,
            mode='multi'
        )
        
        # Benchmark different sizes
        for num_sims in [1000, 10000, 50000]:
            print(f"\nGenerating {num_sims} simulations...")
            key, subkey = jax_random.split(key)
            
            start = time.time()
            theta, x, x_mean, x_std = simulator.generate_training_data(
                subkey, prior_low, prior_high,
                num_simulations=num_sims,
                window_sites=100,
                mode='multi'
            )
            elapsed = time.time() - start
            
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Rate: {num_sims/elapsed:.1f} sims/sec")
            print(f"  Shape: theta={theta.shape}, x={x.shape}")
    
    except ImportError as e:
        print("\n" + "="*80)
        print("JAX SIMULATOR")
        print("="*80)
        print(f"❌ JAX not available: {e}")
        print("\nTo install JAX:")
        print("  CPU only: pip install jax jaxlib")
        print("  GPU: see https://github.com/google/jax#installation")


def compare_output_consistency():
    """Verify that JAX and PyTorch produce similar outputs"""
    print("\n" + "="*80)
    print("OUTPUT CONSISTENCY CHECK")
    print("="*80)
    
    try:
        from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM, create_prior
        from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_inference import generate_likelihood_training_data
        from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator_jax import PatchForagingDDMJax, create_prior_jax
        import jax.random as jax_random
        
        # Generate same data with both
        num_sims = 1000
        
        print(f"\nGenerating {num_sims} samples with each simulator...")
        
        # PyTorch
        torch.manual_seed(42)
        np.random.seed(42)
        simulator_pt = PatchForagingDDM()
        prior_pt = create_prior()
        theta_pt, x_pt, _, _ = generate_likelihood_training_data(
            simulator_pt, prior_pt, num_simulations=num_sims, window_sites=100, mode='multi'
        )
        
        # JAX
        simulator_jax = PatchForagingDDMJax()
        prior_low, prior_high = create_prior_jax()
        key = jax_random.PRNGKey(42)
        theta_jax, x_jax, _, _ = simulator_jax.generate_training_data(
            key, prior_low, prior_high, num_simulations=num_sims, window_sites=100, mode='multi'
        )
        
        # Compare distributions (not exact due to different RNG)
        print("\nComparing output distributions:")
        print(f"  PyTorch x mean: {x_pt.mean(dim=0)}")
        print(f"  JAX x mean:     {x_jax.mean(dim=0)}")
        print(f"\n  PyTorch x std:  {x_pt.std(dim=0)}")
        print(f"  JAX x std:      {x_jax.std(dim=0)}")
        
        # Check if distributions are similar (within ~10%)
        mean_diff = ((x_pt.mean(dim=0) - x_jax.mean(dim=0)).abs() / (x_pt.mean(dim=0) + 1e-6)).mean()
        std_diff = ((x_pt.std(dim=0) - x_jax.std(dim=0)).abs() / (x_pt.std(dim=0) + 1e-6)).mean()
        
        print(f"\n  Mean difference: {mean_diff:.1%}")
        print(f"  Std difference:  {std_diff:.1%}")
        
        if mean_diff < 0.1 and std_diff < 0.1:
            print("\n  ✓ Outputs are consistent!")
        else:
            print("\n  ⚠️  Outputs differ more than expected")
            
    except ImportError:
        print("\n  Skipping (JAX not available)")


if __name__ == "__main__":
    print("="*80)
    print("SIMULATOR BENCHMARK")
    print("="*80)
    
    # Run benchmarks
    benchmark_pytorch()
    benchmark_jax()
    compare_output_consistency()
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print("\nTo use JAX in parameter sweep:")
    print("  python snle_parameter_sweep.py --jax")
    print("  python snle_parameter_sweep.py test --jax")