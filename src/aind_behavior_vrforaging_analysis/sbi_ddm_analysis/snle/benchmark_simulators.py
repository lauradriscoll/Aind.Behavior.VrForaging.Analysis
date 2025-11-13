"""
Benchmark JAX vs PyTorch simulator performance.

Usage:
    python benchmark_simulators.py
"""

# CRITICAL: Set JAX platform BEFORE any imports
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import time
import torch
import numpy as np
from jax import random


def benchmark_pytorch():
    """Benchmark PyTorch simulator"""
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM, create_prior
    from snle_inference import generate_likelihood_training_data
    
    print("\n" + "="*80)
    print("PYTORCH SIMULATOR")
    print("="*80)
    
    simulator = PatchForagingDDM()
    prior = create_prior()
    
    # Warmup
    print("\nWarming up...")
    _, _, _, _ = generate_likelihood_training_data(
        simulator, prior, num_simulations=100, window_sites=100, 
        mode='single', use_jax=False
    )
    
    # Benchmark different sizes
    for num_sims in [1000, 10000]:
        print(f"\nGenerating {num_sims} simulations...")
        start = time.time()
        theta, x, x_mean, x_std = generate_likelihood_training_data(
            simulator, prior, 
            num_simulations=num_sims, 
            window_sites=100, 
            mode='single',
            use_jax=False
        )
        elapsed = time.time() - start
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Rate: {num_sims/elapsed:.1f} sims/sec")
        print(f"  Shape: theta={theta.shape}, x={x.shape}")


def benchmark_jax():
    """Benchmark JAX simulator"""
    try:
        from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator_jax import PatchForagingDDM_JAX, create_prior_jax
        import jax
        
        print("\n" + "="*80)
        print("JAX SIMULATOR")
        print("="*80)
        print(f"JAX devices: {jax.devices()}")
        print(f"JAX backend: {jax.default_backend()}")
        
        simulator = PatchForagingDDM_JAX()
        prior_low, prior_high = create_prior_jax()
        rng_key = random.PRNGKey(42)
        
        # Warmup (important for JAX JIT compilation)
        print("\nWarming up (JIT compiling)...")
        rng_key, subkey = random.split(rng_key)
        theta_warmup, x_warmup = simulator.generate_training_data(
            prior_low, prior_high,
            num_samples=100,
            rng_key=subkey,
            mode='single',
            return_torch=True
        )
        print(f"  Warmup complete. Shape: theta={theta_warmup.shape}, x={x_warmup.shape}")
        
        # Benchmark different sizes
        for num_sims in [1000, 10000, 100000]:
            print(f"\nGenerating {num_sims} simulations...")
            rng_key, subkey = random.split(rng_key)
            
            start = time.time()
            theta, x = simulator.generate_training_data(
                prior_low, prior_high,
                num_samples=num_sims,
                rng_key=subkey,
                mode='single',
                return_torch=True
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
    except Exception as e:
        print(f"\n❌ Error running JAX benchmark: {e}")
        import traceback
        traceback.print_exc()


def benchmark_jax_with_snle_interface():
    """Benchmark JAX simulator using the SNLE interface"""
    try:
        from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator_jax import PatchForagingDDM_JAX
        from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import create_prior
        from snle_inference import generate_likelihood_training_data
        
        print("\n" + "="*80)
        print("JAX SIMULATOR (via SNLE interface)")
        print("="*80)
        
        simulator = PatchForagingDDM_JAX()
        prior = create_prior()
        rng_key = random.PRNGKey(42)
        
        # Warmup
        print("\nWarming up (JIT compiling)...")
        rng_key, subkey = random.split(rng_key)
        _, _, _, _ = generate_likelihood_training_data(
            simulator, prior, num_simulations=100, window_sites=100,
            mode='single', use_jax=True, rng_key=subkey
        )
        
        # Benchmark different sizes
        for num_sims in [1000, 10000, 100000]:
            print(f"\nGenerating {num_sims} simulations...")
            rng_key, subkey = random.split(rng_key)
            
            start = time.time()
            theta, x, x_mean, x_std = generate_likelihood_training_data(
                simulator, prior,
                num_simulations=num_sims,
                window_sites=100,
                mode='single',
                use_jax=True,
                rng_key=subkey
            )
            elapsed = time.time() - start
            
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Rate: {num_sims/elapsed:.1f} sims/sec")
            print(f"  Shape: theta={theta.shape}, x={x.shape}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def compare_output_consistency():
    """Verify that JAX and PyTorch produce similar outputs"""
    print("\n" + "="*80)
    print("OUTPUT CONSISTENCY CHECK")
    print("="*80)
    
    try:
        from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM, create_prior
        from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator_jax import PatchForagingDDM_JAX
        
        num_sims = 1000
        
        print(f"\nGenerating {num_sims} samples with each simulator...")
        
        # PyTorch
        print("\n  Running PyTorch simulator...")
        torch.manual_seed(42)
        np.random.seed(42)
        simulator_pt = PatchForagingDDM()
        prior = create_prior()
        
        theta_pt_list = []
        x_pt_list = []
        for _ in range(num_sims):
            theta = prior.sample()
            param_gen = simulator_pt.evolve_params(mode='walk', theta_init=theta, sigma=0.0)
            _, summary_stats, _ = simulator_pt.simulate_trial(param_gen, 100, return_aggregate=False)
            theta_pt_list.append(theta)
            x_pt_list.append(summary_stats)
        
        theta_pt = torch.stack(theta_pt_list)
        x_pt = torch.stack(x_pt_list)
        
        # JAX
        print("  Running JAX simulator...")
        simulator_jax = PatchForagingDDM_JAX()
        prior_low = prior.base_dist.low.numpy()
        prior_high = prior.base_dist.high.numpy()
        rng_key = random.PRNGKey(42)
        
        theta_jax, x_jax = simulator_jax.generate_training_data(
            prior_low, prior_high, num_sims, rng_key, mode='single', return_torch=True
        )
        
        # Compare distributions (not exact due to different RNG)
        print("\n  Comparing output distributions:")
        print(f"    PyTorch theta mean: {theta_pt.mean(dim=0)}")
        print(f"    JAX theta mean:     {theta_jax.mean(dim=0)}")
        print(f"\n    PyTorch x mean: {x_pt.mean(dim=0)}")
        print(f"    JAX x mean:     {x_jax.mean(dim=0)}")
        print(f"\n    PyTorch x std:  {x_pt.std(dim=0)}")
        print(f"    JAX x std:      {x_jax.std(dim=0)}")
        
        # Check if distributions are similar (within ~20% - different RNG will cause variation)
        mean_diff = ((x_pt.mean(dim=0) - x_jax.mean(dim=0)).abs() / (x_pt.mean(dim=0) + 1e-6)).mean()
        std_diff = ((x_pt.std(dim=0) - x_jax.std(dim=0)).abs() / (x_pt.std(dim=0) + 1e-6)).mean()
        
        print(f"\n    Mean difference: {mean_diff:.1%}")
        print(f"    Std difference:  {std_diff:.1%}")
        
        if mean_diff < 0.2 and std_diff < 0.2:
            print("\n    ✓ Outputs are reasonably consistent!")
            print("      (Small differences expected due to different RNG implementations)")
        else:
            print("\n    ⚠️  Outputs differ more than expected")
            print("      This could indicate a bug or implementation difference")
            
    except ImportError as e:
        print(f"\n  ❌ Skipping (module not available): {e}")
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        import traceback
        traceback.print_exc()


def print_speedup_summary():
    """Print summary of expected speedups"""
    print("\n" + "="*80)
    print("EXPECTED SPEEDUP SUMMARY")
    print("="*80)
    print("\nFor 1M simulations:")
    print("  PyTorch: ~10-100 sims/sec  → ~10,000 seconds (~3 hours)")
    print("  JAX:     ~1,000-10,000/sec → ~100-1,000 seconds (2-17 minutes)")
    print("\n  Expected speedup: 10-100x")
    print("\nNote: Actual performance depends on:")
    print("  - CPU/GPU hardware")
    print("  - Patch complexity (number of sites per patch)")
    print("  - JIT compilation overhead (first run is slower)")
    print("="*80)


if __name__ == "__main__":
    print("="*80)
    print("SIMULATOR BENCHMARK")
    print("="*80)
    print("\nThis benchmark compares PyTorch and JAX simulators.")
    print("JAX uses CPU backend on Apple Silicon (avoids Metal issues).")
    
    # Run benchmarks
    benchmark_pytorch()
    benchmark_jax()
    benchmark_jax_with_snle_interface()
    compare_output_consistency()
    print_speedup_summary()
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print("\nTo use JAX for training:")
    print("  See example_jax_snle.py for usage")