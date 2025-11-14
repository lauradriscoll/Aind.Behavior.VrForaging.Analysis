"""
SNLE inference for patch foraging.

Purpose: Train neural likelihood estimator and run MCMC inference for patch-level parameters.
Supports both single-patch and multi-patch inference modes.
"""

import os
# Force CPU backend on Apple Silicon to avoid Metal issues
os.environ['JAX_PLATFORMS'] = 'cpu'

import torch
import numpy as np
from tqdm import tqdm
from sbi.inference import SNLE
from jax import random
import jax.numpy as jnp
rng_key = random.PRNGKey(0)
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_utils import save_snle_model, load_snle_model

def generate_likelihood_training_data(simulator, rng_key, prior_low, prior_high, num_samples=10000):
    """
    Generate training data for SNLE from the simulator.

    Args:
        simulator: JAX simulator object
        rng_key: JAX PRNGKey
        prior_low: lower bounds of prior (array-like, shape [4])
        prior_high: upper bounds of prior (array-like, shape [4])
        num_samples: number of training samples

    Returns:
        theta_samples_jax, x_samples_jax, x_mean, x_std, rng_key
    """
    # Generate theta and x samples
    prior_low_jax = jnp.array(prior_low)
    prior_high_jax = jnp.array(prior_high)

    theta_samples_jax, x_samples_jax = simulator.generate_training_data(
        num_samples=num_samples,
        rng_key=rng_key,
        prior_low=prior_low_jax,
        prior_high=prior_high_jax
    )

    theta_samples = torch.tensor(theta_samples_jax)
    x_samples = torch.tensor(x_samples_jax)

    # Compute mean and std for normalization
    x_mean = torch.mean(x_samples, dim=0)
    x_std = torch.std(x_samples, dim=0)

    return theta_samples, x_samples, x_mean, x_std, rng_key

def train_snle(simulator, prior, mode='multi', max_num_epochs=50, batch_size=50, num_samples=10000,
               rng_key=None, prior_low=[0.01, 0.01, 0.0, 0.0], 
               prior_high=[1.5, 1.5, 1.5, 0.1]):

    """
    Train SNLE to learn p(summary_stats | theta) using a JAX simulator.
    
    Returns:
        likelihood_estimator: Trained SNLE
        inference: SNLE inference object
        x_mean: Mean of training data
        x_std: Std of training data
        training_history: Dict with training history
        rng_key: Updated JAX RNG key
    """

    # --- 1. Generate training data ---
    theta_samples, x_samples, x_mean, x_std, rng_key = generate_likelihood_training_data(
    simulator,
    num_samples=num_samples,
    rng_key=rng_key,
    prior_low=prior_low,
    prior_high=prior_high
)
    # --- 2. Initialize SNLE ---
    print(f"\nTraining SNLE ({mode} mode)...")
    inference = SNLE(prior=prior)

    # --- 3. Append simulations ---

    inference.append_simulations(theta_samples, x_samples)

    # --- 4. Train likelihood estimator ---
    likelihood_estimator = inference.train(
        training_batch_size=batch_size,
        max_num_epochs=max_num_epochs,
        show_train_summary=True,
        stop_after_epochs=20,  # early stopping patience
    )

    # --- 5. Extract training history safely ---
    summary = getattr(inference, "_summary", {})
    best_val = summary.get('best_validation_loss', None)
    if isinstance(best_val, list):
        best_val = best_val[0] if len(best_val) > 0 else None

    epochs_trained = summary.get('epochs_trained', len(summary.get('training_loss', [])))
    if isinstance(epochs_trained, list):
        epochs_trained = epochs_trained[0] if len(epochs_trained) > 0 else len(summary.get('training_loss', []))

    training_history = {
        'train_loss': summary.get('training_loss', []),
        'val_loss': summary.get('validation_loss', []),
        'epochs': list(range(len(summary.get('training_loss', [])))),
        'best_val_loss': best_val,
        'epochs_trained': epochs_trained,
    }

    print(f"\nSNLE training complete ({mode} mode)!")
    print(f"Epochs trained: {training_history['epochs_trained']}")
    if training_history['train_loss']:
        print(f"Final train loss: {training_history['train_loss'][-1]:.4f}")
    if training_history['val_loss']:
        print(f"Final validation loss: {training_history['val_loss'][-1]:.4f}")
    if best_val is not None:
        print(f"Best validation loss: {best_val:.4f}")

    return likelihood_estimator, inference, x_mean, x_std, training_history, rng_key

def infer_parameters_snle(inference, observed_stats, 
                          x_mean, x_std,
                          num_samples=1000, warmup_steps=200,
                          mcmc_method="slice_np_vectorized"):
    """
    Infer parameters from observed summary statistics using MCMC.
    Handles JAX or PyTorch inputs safely.
    """

    # Flatten / ensure 1D
    observed_stats = observed_stats.view(-1) if observed_stats.ndim == 0 else observed_stats

    # Normalize using training mean/std
    observed_stats_normalized = (observed_stats - x_mean.squeeze()) / (x_std.squeeze() + 1e-8)

    # Build posterior using the inference object's method
    posterior = inference.build_posterior(
        sample_with="mcmc",
        mcmc_method=mcmc_method,
        mcmc_parameters={
            "num_chains": 1,
            "thin": 5,
            "warmup_steps": warmup_steps,
        }
    )
    
    # Sample from posterior
    posterior_samples = posterior.sample(
        (num_samples,), 
        x=observed_stats_normalized,
        show_progress_bars=True
    )
    
    return posterior_samples



# Test
if __name__ == "__main__":
    import torch
    from jax import random
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator_jax import PatchForagingDDM_JAX
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import create_prior
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_inference import (
        train_snle,
        infer_parameters_snle,
        save_snle_model,
        load_snle_model
    )

    print("Testing SNLE inference module...")

    # --- Setup ---
    num_window_sites = 100
    simulator = PatchForagingDDM_JAX(max_sites_per_window=num_window_sites)
        # Get prior bounds for JAX simulator
    prior, prior_low, prior_high = create_prior()
    rng_key = random.PRNGKey(0)  # Initialize RNG once

    # --- Multi-patch training ---
    print("\n1. Training multi-patch SNLE model...")
    likelihood_estimator, inference_multi, x_mean_multi, x_std_multi, history_multi, rng_key = train_snle(
        simulator, prior=prior, prior_low=prior_low, prior_high=prior_high,
        mode='multi', max_num_epochs=10, batch_size=50, rng_key=rng_key
    )
    print(f"   Multi-patch model trained: {history_multi['epochs_trained']} epochs")

    # --- Simulate one window for inference ---
    true_theta = torch.tensor([0.2, 0.8, 0.3, 0.05], dtype=torch.float32)
    rng_key, subkey = random.split(rng_key)
    _, observed_stats_jax = simulator.simulate_one_window(true_theta, subkey)

    # Convert JAX array to PyTorch tensor
    observed_stats = torch.tensor(np.array(observed_stats_jax), dtype=torch.float32)

    # --- Parameter inference ---
    print("\n2. Testing inference...")
    posterior_samples = infer_parameters_snle(
        inference_multi,
        observed_stats,
        x_mean_multi,
        x_std_multi,
        num_samples=100,
        warmup_steps=50
    )
    print(f"   Posterior samples shape: {posterior_samples.shape}")
    print(f"   True theta:      {true_theta}")
    print(f"   Posterior mean:  {posterior_samples.mean(dim=0)}")

    # --- Save / load test ---
    print("\n3. Testing save/load...")
    model_dir = save_snle_model(inference_multi, x_mean_multi, x_std_multi, mode='multi')
    loaded_inference, loaded_mean, loaded_std, loaded_mode, analysis_dir = load_snle_model(model_dir)
    print("   Model saved and loaded successfully")
    print(f"   Analysis directory: {analysis_dir}")

    print("\n✓ All tests passed!")
