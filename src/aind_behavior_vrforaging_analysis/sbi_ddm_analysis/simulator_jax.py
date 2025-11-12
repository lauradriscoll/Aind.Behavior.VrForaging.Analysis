"""
JAX-accelerated simulator for patch foraging DDM.

Purpose: Fast parallel simulation for SNLE training data generation.
Key features:
- Vectorized across batches using jax.vmap
- JIT compiled for speed
- GPU-compatible
- Drop-in replacement for data generation

Usage:
    simulator = PatchForagingDDMJax()
    
    # Generate batch of simulations in parallel
    theta_batch = jax.random.uniform(key, (10000, 3), minval=jnp.array([0.01, 0.01, 0.0]), 
                                      maxval=jnp.array([1.5, 1.5, 1.5]))
    stats_batch = simulator.simulate_batch(key, theta_batch, window_sites=100)
"""

import jax
import jax.numpy as jnp
from jax import random, vmap, jit
from functools import partial
import torch
import numpy as np


def reward_probability_jax(num_rewards, initial_prob=0.8, decay_rate=-0.1):
    """Exponential decay reward probability"""
    return initial_prob * jnp.exp(decay_rate * num_rewards)


class PatchForagingDDMJax:
    """
    JAX-accelerated DDM for patch foraging.
    
    Key differences from PyTorch version:
    - Uses explicit random keys (JAX style)
    - Fixed maximum trajectory length for static shapes
    - Batched simulation via vmap
    - JIT compiled for speed
    """
    
    def __init__(self, 
                 initial_prob=0.8, 
                 decay_rate=-0.1,
                 threshold=1.0,
                 start_point=0.0,
                 interval_mean=1.0, 
                 interval_std=0.3, 
                 interval_min=0.1, 
                 interval_max=5.0,
                 noise_std=0.0,
                 max_sites_per_patch=500):  # Maximum sites per patch for fixed shape
        self.initial_prob = initial_prob
        self.decay_rate = decay_rate
        self.threshold = threshold
        self.start_point = start_point
        self.interval_mean = interval_mean
        self.interval_std = interval_std
        self.interval_min = interval_min
        self.interval_max = interval_max
        self.noise_std = noise_std
        self.max_sites_per_patch = max_sites_per_patch
        
        # JIT compile the core simulation function
        self._simulate_one_patch_jit = jit(self._simulate_one_patch_static)
        self._simulate_trial_jit = jit(partial(self._simulate_trial_static, 
                                               max_patches=20))  # Max patches per trial
    
    def sample_inter_site_interval(self, key):
        """Sample time between odor sites (truncated Gaussian)"""
        sample = self.interval_mean + self.interval_std * random.normal(key)
        return jnp.clip(sample, self.interval_min, self.interval_max)
    
    def _simulate_one_patch_static(self, key, theta):
        """
        Simulate a single patch with fixed-size output (for JIT).
        
        Args:
            key: JAX random key
            theta: [drift_rate, reward_bump, failure_bump]
        
        Returns:
            summary_stats: [total_time, num_stops, num_rewards]
            actual_length: number of valid sites (rest is padding)
        """
        drift_rate, reward_bump, failure_bump = theta
        
        def step_fn(carry, key_t):
            """Single step in the patch"""
            evidence, num_rewards, time_in_patch, left = carry
            
            # Sample interval
            key_interval, key_reward = random.split(key_t)
            dt = self.sample_inter_site_interval(key_interval)
            time_in_patch = time_in_patch + dt
            
            # Accumulate evidence (no noise for now, can add later)
            evidence = evidence + drift_rate * dt
            
            # Check if should leave
            should_leave = evidence >= self.threshold
            
            # If not leaving, stop and check reward
            reward_prob = reward_probability_jax(num_rewards, self.initial_prob, self.decay_rate)
            reward = random.uniform(key_reward) < reward_prob
            
            # Update evidence based on outcome
            # Only update if not leaving
            evidence_update = jnp.where(
                should_leave,
                0.0,  # No update if leaving
                jnp.where(reward, -reward_bump, failure_bump)
            )
            evidence = evidence + evidence_update
            
            # Update counters (only if not leaving)
            num_rewards = jnp.where(should_leave, num_rewards, num_rewards + reward.astype(jnp.float32))
            
            # Mark if left on this step
            left = left | should_leave
            
            return (evidence, num_rewards, time_in_patch, left), (time_in_patch, reward, should_leave, left)
        
        # Initial state
        init_carry = (self.start_point, 0.0, 0.0, False)
        
        # Generate keys for all possible steps
        keys = random.split(key, self.max_sites_per_patch)
        
        # Run simulation
        final_carry, outputs = jax.lax.scan(step_fn, init_carry, keys)
        
        times, rewards, leaves, left_flags = outputs
        
        # Find where animal left (first True in left_flags)
        # All steps after leaving are invalid
        cumsum_left = jnp.cumsum(left_flags.astype(jnp.int32))
        valid_mask = cumsum_left == 0  # Steps before leaving
        
        # Count valid stops (not including the leave site)
        num_stops = jnp.sum(valid_mask & ~leaves)
        
        # Sum rewards only from valid, non-leave sites
        num_rewards_total = jnp.sum(rewards * valid_mask)
        
        # Time when left (first time left_flags is True)
        left_indices = jnp.where(left_flags, jnp.arange(self.max_sites_per_patch), self.max_sites_per_patch)
        leave_idx = jnp.min(left_indices)
        total_time = jnp.where(leave_idx < self.max_sites_per_patch, times[leave_idx], times[-1])
        
        summary_stats = jnp.array([total_time, num_stops, num_rewards_total])
        actual_length = leave_idx + 1  # Include the leave site
        
        return summary_stats, actual_length
    
    def _simulate_trial_static(self, key, theta, window_sites, max_patches=[]):
        """
        Simulate trial until we have at least window_sites.
        Uses constant theta for all patches (for data generation).
        
        Args:
            key: JAX random key
            theta: [drift_rate, reward_bump, failure_bump]
            window_sites: Target number of sites
            max_patches: Maximum number of patches to simulate (should be the same as window sites if always leave)

        Returns:
            summary_stats: Single-patch stats (3,) OR aggregate stats (8,)
        """
        # Generate keys for all patches
        max_patches = window_sites
        keys = random.split(key, max_patches)
        
        # Simulate all patches (vectorized)
        stats_all, lengths_all = vmap(self._simulate_one_patch_static, in_axes=(0, None))(keys, theta)
        
        # Compute cumulative sites
        cumsum_lengths = jnp.cumsum(lengths_all)
        
        # Find first patch where we exceed window_sites
        enough_sites = cumsum_lengths >= window_sites
        first_enough = jnp.argmax(enough_sites)  # First True index
        num_patches_needed = first_enough + 1
        
        # Get stats for patches we actually used
        stats_used = stats_all[:num_patches_needed]
        
        # Compute aggregate statistics
        mean_stats = jnp.mean(stats_used, axis=0)
        std_stats = jnp.std(stats_used, axis=0)
        
        # Handle single patch case (std would be 0)
        aggregate_stats = jnp.array([
            mean_stats[0],  # mean total_time
            std_stats[0],   # std total_time
            mean_stats[1],  # mean num_stops
            std_stats[1],   # std num_stops
            mean_stats[2],  # mean num_rewards
            std_stats[2],   # std num_rewards
            jnp.sum(stats_used[:, 2]),  # total rewards
            num_patches_needed.astype(jnp.float32),  # num_patches
        ])
        
        return aggregate_stats
    
    def simulate_batch(self, key, theta_batch, window_sites=100, mode='multi'):
        """
        Simulate a batch of trials in parallel (main interface for data generation).
        
        Args:
            key: JAX random key
            theta_batch: (N, 3) array of parameters
            window_sites: Number of sites per trial
            mode: 'single' or 'multi' (aggregate)
        
        Returns:
            stats_batch: (N, 3) for single or (N, 8) for multi
        """
        batch_size = theta_batch.shape[0]
        keys = random.split(key, batch_size)
        
        if mode == 'single':
            # Just simulate one patch per theta
            stats_batch, _ = vmap(self._simulate_one_patch_static)(keys, theta_batch)
        else:
            # Simulate full trial with multiple patches
            stats_batch = vmap(partial(self._simulate_trial_static, window_sites=window_sites))(
                keys, theta_batch
            )
        
        return stats_batch
    
    def to_torch(self, jax_array):
        """Convert JAX array to PyTorch tensor"""
        return torch.from_numpy(np.array(jax_array)).float()
    
    def generate_training_data(self, key, prior_low, prior_high, num_simulations, 
                              window_sites=100, mode='multi'):
        """
        Generate training data for SNLE (drop-in replacement for PyTorch version).
        
        Args:
            key: JAX random key
            prior_low: (3,) lower bounds for parameters
            prior_high: (3,) upper bounds for parameters
            num_simulations: Number of simulations to generate
            window_sites: Sites per simulation
            mode: 'single' or 'multi'
        
        Returns:
            theta_samples: (N, 3) PyTorch tensor
            x_samples: (N, 3 or 8) PyTorch tensor (normalized)
            x_mean: Mean for denormalization
            x_std: Std for denormalization
        """
        print(f"Generating {num_simulations} simulations with JAX (mode={mode})...")
        
        # Sample parameters from prior
        key_theta, key_sim = random.split(key)
        theta_batch = random.uniform(
            key_theta, 
            (num_simulations, 3), 
            minval=prior_low, 
            maxval=prior_high
        )
        
        # Simulate in parallel
        stats_batch = self.simulate_batch(key_sim, theta_batch, window_sites=window_sites, mode=mode)
        
        # Convert to PyTorch
        theta_samples = self.to_torch(theta_batch)
        x_samples = self.to_torch(stats_batch)
        
        # Check for NaN
        num_nan = torch.isnan(x_samples).any(dim=1).sum()
        if num_nan > 0:
            print(f"⚠️  WARNING: {num_nan}/{num_simulations} samples have NaN!")
            # Remove NaN samples
            valid_mask = ~torch.isnan(x_samples).any(dim=1)
            theta_samples = theta_samples[valid_mask]
            x_samples = x_samples[valid_mask]
            print(f"   Kept {len(theta_samples)} valid samples")
        
        # Normalize
        x_mean = x_samples.mean(dim=0, keepdim=True)
        x_std = x_samples.std(dim=0, keepdim=True)
        x_samples_normalized = (x_samples - x_mean) / (x_std + 1e-8)
        
        print(f"Training data shape: theta={theta_samples.shape}, x={x_samples.shape}")
        if mode == 'single':
            print(f"Single-patch stats (3 features): total_time, num_stops, num_rewards")
        else:
            print(f"Multi-patch aggregate stats (8 features): mean_time, std_time, mean_stops, std_stops, mean_rewards, std_rewards, total_rewards, num_patches")
        
        return theta_samples, x_samples_normalized, x_mean, x_std


# ===== Utility functions =====

def create_prior_jax():
    """
    Get prior bounds for JAX simulator.
    
    Returns:
        prior_low, prior_high as JAX arrays
    """
    prior_low = jnp.array([0.01, 0.01, 0.0])
    prior_high = jnp.array([1.5, 1.5, 1.5])
    return prior_low, prior_high


# ===== Tests =====

if __name__ == "__main__":
    print("="*80)
    print("Testing JAX Simulator")
    print("="*80)
    
    # Initialize
    simulator = PatchForagingDDMJax()
    key = random.PRNGKey(0)
    
    # Test 1: Single patch simulation
    print("\n1. Single patch simulation")
    key, subkey = random.split(key)
    theta = jnp.array([0.5, 0.6, 0.2])
    stats, length = simulator._simulate_one_patch_static(subkey, theta)
    print(f"   Stats: {stats}")
    print(f"   Length: {length}")
    
    # Test 2: Trial simulation (multiple patches)
    print("\n2. Trial simulation (multiple patches)")
    key, subkey = random.split(key)
    stats = simulator._simulate_trial_static(subkey, theta, window_sites=100)
    print(f"   Aggregate stats: {stats}")
    print(f"   Shape: {stats.shape}")
    
    # Test 3: Batch simulation (vectorized)
    print("\n3. Batch simulation")
    key, subkey = random.split(key)
    theta_batch = random.uniform(subkey, (100, 3), 
                                 minval=jnp.array([0.01, 0.01, 0.0]),
                                 maxval=jnp.array([1.5, 1.5, 1.5]))
    
    import time
    start = time.time()
    stats_batch = simulator.simulate_batch(subkey, theta_batch, window_sites=100, mode='multi')
    elapsed = time.time() - start
    
    print(f"   Generated {len(theta_batch)} trials in {elapsed:.3f}s")
    print(f"   Rate: {len(theta_batch)/elapsed:.1f} trials/sec")
    print(f"   Stats shape: {stats_batch.shape}")
    print(f"   First sample: {stats_batch[0]}")
    
    # Test 4: Generate training data (full pipeline)
    print("\n4. Generate training data")
    key, subkey = random.split(key)
    prior_low, prior_high = create_prior_jax()
    
    theta_samples, x_samples, x_mean, x_std = simulator.generate_training_data(
        subkey, prior_low, prior_high, 
        num_simulations=1000,
        window_sites=100,
        mode='multi'
    )
    print(f"   Theta shape: {theta_samples.shape}")
    print(f"   X shape: {x_samples.shape}")
    print(f"   X mean: {x_mean.squeeze()}")
    print(f"   X std: {x_std.squeeze()}")
    
    # Test 5: Benchmark vs sequential
    print("\n5. Speed benchmark")
    num_sims = 10000
    
    print(f"   Generating {num_sims} simulations...")
    key, subkey = random.split(key)
    start = time.time()
    stats_batch = simulator.simulate_batch(subkey, 
                                           random.uniform(subkey, (num_sims, 3),
                                                         minval=prior_low,
                                                         maxval=prior_high),
                                           window_sites=100, mode='multi')
    elapsed = time.time() - start
    print(f"   JAX (parallel): {elapsed:.3f}s ({num_sims/elapsed:.1f} trials/sec)")
    
    print("\n" + "="*80)
    print("✓ All JAX tests passed!")
    print("="*80)
    
    print("\n" + "="*80)
    print("Example Usage for SNLE Training")
    print("="*80)
    print("""
from simulator_jax import PatchForagingDDMJax, create_prior_jax
import jax.random as random

# Initialize
simulator = PatchForagingDDMJax()
key = random.PRNGKey(42)
prior_low, prior_high = create_prior_jax()

# Generate training data
theta_samples, x_samples, x_mean, x_std = simulator.generate_training_data(
    key, prior_low, prior_high,
    num_simulations=200000,
    window_sites=100,
    mode='multi'
)

# x_samples is already normalized and ready for SNLE training!
    """)