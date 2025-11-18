"""
JAX-based simulator for patch foraging DDM.
4 parameters: drift_rate, reward_bump, failure_bump, noise_std
7 summary statistics: max time, mean time, std time, mean stops, std stops, mean rewards, std rewards

"""

import os
# Force CPU backend on Apple Silicon to avoid Metal issues
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
from jax import random, jit, vmap

from tensorflow_probability.substrates.jax import distributions as tfd


def reward_probability(num_rewards, initial_prob=0.8, decay_rate=-0.1):
    """Exponential decay reward probability based on number of rewards collected"""
    return initial_prob * jnp.exp(decay_rate * num_rewards)


class PatchForagingDDM_JAX:
    """
    JAX implementation of DDM for patch foraging.
    Simulates patch foraging behavior with evidence accumulation, rewards, and patch leaving decisions.
    4 parameters:
        - drift_rate: evidence accumulation rate
        - reward_bump: evidence boost from receiving reward
        - failure_bump: evidence boost from not receiving reward
        - noise_std: standard deviation of noise in evidence accumulation
    7 summary statistics:
        - max patch time
        - mean patch time       
        - std patch time
        - mean stops
        - std stops
        - mean rewards
        - std rewards
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
                 max_sites_per_window=500):  # For pre-allocation
        self.initial_prob = initial_prob
        self.decay_rate = decay_rate
        self.threshold = threshold
        self.start_point = start_point
        self.interval_mean = interval_mean
        self.interval_std = interval_std
        self.interval_min = interval_min
        self.interval_max = interval_max
        self.max_sites_per_window = max_sites_per_window
        
        # JIT compile the core simulation function
        self._simulate_one_window_jit = jit(self._simulate_one_window_core)
    
    def _simulate_one_window_core(self, theta, rng_key):
        """
        Core JIT-compiled function to simulate one window.
        Uses jax.lax.while_loop for efficient compilation.
        
        Args:
            theta: (4,) array [drift_rate, reward_bump, failure_bump, noise_std]
            rng_key: JAX random key
            
        Returns:
            window_data: (max_sites, 3) array [time_in_patch, reward, stopped]
            summary_stats: (7,) array (max patch time, mean patch time, 
                                        std patch time, mean stops, std stops, 
                                        mean rewards, std rewards)
        """
        drift_rate, reward_bump, failure_bump, noise_std = theta
        
        # Pre-allocate arrays
        window_data = jnp.zeros((self.max_sites_per_window, 3))
        
        # Split RNG keys for different random operations
        key_intervals, key_noise, key_rewards = random.split(rng_key, 3)
        
        # Pre-generate all random numbers (faster than generating in loop)
        intervals = random.truncated_normal(
            key_intervals, 
            lower=(self.interval_min - self.interval_mean) / self.interval_std,
            upper=(self.interval_max - self.interval_mean) / self.interval_std,
            shape=(self.max_sites_per_window,)
        ) * self.interval_std + self.interval_mean
        
        noise_samples = random.normal(key_noise, shape=(self.max_sites_per_window,))
        reward_samples = random.uniform(key_rewards, shape=(self.max_sites_per_window,))
        # State tuple for while loop: (evidence, num_rewards, site_idx, global_time, window_data)
        def cond_fn(state):
            evidence, num_rewards, site_idx, global_time, patch_time, window_data = state
            return (site_idx < self.max_sites_per_window)
        
        def body_fn(state):
            evidence, num_rewards, site_idx, global_time, patch_time, window_data = state
            
            # Get pre-generated random values for this site
            dt = intervals[site_idx]
            noise = noise_samples[site_idx]
            reward_sample = reward_samples[site_idx]
            
            # Update time and evidence
            global_time = global_time + dt
            patch_time = patch_time + dt  # Time spent in current patch
            evidence = evidence + drift_rate * dt
            evidence = jnp.where(
                noise_std > 0,
                evidence + noise_std * noise * jnp.sqrt(dt),
                evidence
            )
            
            # Check if we should leave
            should_leave = evidence >= self.threshold
            
            # If not leaving, check for reward
            reward_prob = reward_probability(num_rewards, self.initial_prob, self.decay_rate)
            reward = jnp.where(should_leave, 0, (reward_sample < reward_prob).astype(jnp.float32))
            stopped = jnp.where(should_leave, 0, 1)
            
            # Store data
            window_data = window_data.at[site_idx].set(jnp.array([patch_time, reward, stopped]))

            # Update evidence based on outcome
            evidence = jnp.where(
                should_leave,
                self.start_point,  # reset evidence if leaving
                evidence + jnp.where(reward > 0, -reward_bump, failure_bump)
            )
            
            # Update state
            num_rewards = jnp.where(should_leave, 0, num_rewards + reward)  # reset if leaving
            patch_time = jnp.where(should_leave, 0.0, patch_time)  # reset patch time if leaving
            
            return (evidence, num_rewards, site_idx + 1, global_time, patch_time, window_data)
        
        # Initial state
        init_state = (
            jnp.array(self.start_point),  # evidence
            jnp.array(0.0),                # num_rewards
            jnp.array(0),                  # site_idx
            jnp.array(0.0),                # global_time
            jnp.array(0.0),                # patch_time
            window_data                     # window_data array
        )
        
        # Run simulation
        final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
        _, _, _, _, _, window_data = final_state
        
        # Compute number of stops
        num_stops = jnp.sum(window_data[:, 2])

        def single_patch_case(_):
            return jnp.array([
                jnp.max(window_data[:, 0]),   # total_time in patch max
                jnp.mean(window_data[:, 0]),  # total_time in patch mean
                0.0,                          # std total_time
                jnp.mean(window_data[:, 2]),  # num_stops mean
                0.0,                          # std num_stops
                jnp.mean(window_data[:, 1]),  # num_rewards mean
                0.0,                          # std num_rewards
            ], dtype=jnp.float32)

        def multi_patch_case(_):
            return jnp.array([
                jnp.max(window_data[:, 0]),
                jnp.mean(window_data[:, 0]),
                jnp.std(window_data[:, 0]),
                jnp.mean(window_data[:, 2]),
                jnp.std(window_data[:, 2]),
                jnp.mean(window_data[:, 1]),
                jnp.std(window_data[:, 1]),
            ], dtype=jnp.float32)

        summary_stats = jax.lax.cond(num_stops < 2, single_patch_case, multi_patch_case, operand=None)
    
        return window_data, summary_stats
    
    def simulate_one_window(self, theta, rng_key):
        """
        Simulate one window (user-facing API).
        
        Args:
            theta: (4,) array or list [drift_rate, reward_bump, failure_bump, noise_std]
            rng_key: JAX random key
            
        Returns:
            window_data: (num_sites, 3) array [time_in_patch, reward, stopped]
            summary_stats: (7,) array (max patch time, mean patch time, 
                                        std patch time, mean stops, std stops, 
                                        mean rewards, std rewards)
        """
        theta = jnp.array(theta)
        window_data, summary_stats = self._simulate_one_window_jit(theta, rng_key)
        
        return window_data, summary_stats
    
    # --- Define simulator function matching sbijax API ---
    def simulator_fn(self, theta, seed):
        """
        Simulator function for sbijax.
        Args:
            seed: JAX random key
            theta: dict with key 'theta' containing (n_batch, 4) array of parameters
        Returns:
            x: (n_batch, 7) array of summary statistics
        """
        # Extract theta array from dictionary
        if isinstance(theta, dict):
            theta_array = theta['theta']
        else:
            theta_array = theta
        
        # Get batch size
        if theta_array.ndim == 1:
            theta_array = theta_array.reshape(1, -1)
        n_batch = theta_array.shape[0]
        
        # Generate random keys for each sample
        keys = random.split(seed, n_batch)
        
        # Simulate each sample
        def simulate_one(key, theta_single):
            _, summary_stats = self.simulate_one_window(theta_single, key)
            return summary_stats
        
        # Use vmap to vectorize over batch
        x = vmap(simulate_one)(keys, theta_array)
        
        return x


def create_prior(prior_low=None, prior_high=None):

    if prior_low is None or prior_high is None:
        prior_low  = jnp.array([0.0, 0.0, 0.0, 0.0])
        prior_high = jnp.array([1.0,  1.0,  1.0,  0.5])

    prior_low  = jnp.array(prior_low)
    prior_high = jnp.array(prior_high)

    def prior_fn():
        return tfd.JointDistributionNamed(
            dict(
                theta = tfd.Independent(
                    tfd.Uniform(low=prior_low, high=prior_high),
                    reinterpreted_batch_ndims=1
                )
            ),
            batch_ndims=0
        )

    return prior_fn

# ===== Tests =====
if __name__ == "__main__":
    # Simple test of simulator
    simulator = PatchForagingDDM_JAX()
    rng_key = random.PRNGKey(42)
    theta = jnp.array([0.5, 0.5, 0.3, 0.005])
    
    window_data, summary_stats = simulator.simulate_one_window(theta, rng_key)
    print("Window Data (first 10 sites):")
    print(window_data[:10])
    print("Summary Stats:")
    print(summary_stats)