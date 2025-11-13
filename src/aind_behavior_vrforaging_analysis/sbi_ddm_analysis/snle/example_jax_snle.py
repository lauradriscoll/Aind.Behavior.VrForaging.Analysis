"""
Example: Training SNLE with JAX simulator for fast data generation.

This demonstrates the speed difference between PyTorch and JAX simulators.
"""

# CRITICAL: Set JAX platform BEFORE any JAX imports
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import torch
from sbi.utils.torchutils import BoxUniform

# Now we can import JAX-related modules
from jax import random
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator_jax import PatchForagingDDM_JAX, create_prior_jax
from snle_inference import train_snle

# Setup
print("="*60)
print("SNLE Training with JAX Simulator")
print("="*60)

# Create JAX simulator
simulator_jax = PatchForagingDDM_JAX()

# Create prior (PyTorch BoxUniform for SBI compatibility)
prior = BoxUniform(
    low=torch.tensor([0.01, 0.01, 0.0, 0.0]),
    high=torch.tensor([1.5, 1.5, 1.5, 0.1])
)

# Generate JAX random key
rng_key = random.PRNGKey(42)

# Train SNLE with JAX simulator (FAST!)
print("\n" + "="*60)
print("Training with JAX simulator (fast data generation)")
print("="*60)

likelihood_estimator, inference, x_mean, x_std, history = train_snle(
    simulator=simulator_jax,
    prior=prior,
    num_simulations=10000,  # 10K simulations in seconds!
    mode='single',
    max_num_epochs=50,
    batch_size=50,
    use_jax=True,
    rng_key=rng_key
)

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"Final training loss: {history['train_loss'][-1]:.4f}")
print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
print(f"Epochs trained: {history['epochs_trained']}")

# Example: How to scale up to 1M simulations
print("\n" + "="*60)
print("For 1M simulations, simply change num_simulations:")
print("="*60)
print("""
# This will complete in minutes instead of days:
likelihood_estimator, inference, x_mean, x_std, history = train_snle(
    simulator=simulator_jax,
    prior=prior,
    num_simulations=1_000_000,  # 1 MILLION simulations!
    mode='single',
    max_num_epochs=50,
    batch_size=50,
    use_jax=True,
    rng_key=rng_key
)
""")

print("\n" + "="*60)
print("Speed Comparison:")
print("="*60)
print("PyTorch simulator:  ~1-10 simulations/second   → SLOW")
print("JAX simulator:      ~1,000-10,000 sims/second  → FAST!")
print("="*60)