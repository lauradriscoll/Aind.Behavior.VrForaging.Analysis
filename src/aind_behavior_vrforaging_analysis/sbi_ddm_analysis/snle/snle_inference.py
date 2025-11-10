"""
SNLE inference for patch foraging.

Purpose: Train neural likelihood estimator and run MCMC inference for patch-level parameters.
Supports both single-patch and multi-patch inference modes.
"""

import torch
from tqdm import tqdm
from sbi.inference import SNLE
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_utils import save_snle_model, load_snle_model

def generate_likelihood_training_data(simulator, prior, num_simulations=10000, 
                                     window_sites=100, mode='single'):
    """
    Generate training data for SNLE.
    
    Args:
        simulator: PatchForagingDDM instance
        prior: Prior distribution over parameters
        num_simulations: Number of training samples
        window_sites: Number of sites to simulate
        mode: 'single' for single-patch stats (3 features) 
              'multi' for multi-patch aggregate stats (8 features)
    
    Returns:
        theta_samples: (N, 3) parameters
        x_samples: (N, 3 or 8) summary statistics (normalized)
        x_mean: Mean for denormalization
        x_std: Std for denormalization
    """
    print(f"Generating {num_simulations} training samples (mode={mode})...")
    
    theta_samples = []
    x_samples = []
    
    return_aggregate = (mode == 'multi')
    
    for i in tqdm(range(num_simulations)):
        # Sample parameters
        theta = prior.sample()
        
        # Create constant parameter generator
        param_gen = simulator.walk_params(theta)
        
        # Simulate trial and get stats
        _, summary_stats = simulator.simulate_trial(
            param_gen, 
            window_sites, 
            return_aggregate=return_aggregate
        )
        
        # Debug: Check for NaN on first iteration
        if i == 0:
            print(f"\nFirst sample check:")
            print(f"  theta: {theta}")
            print(f"  summary_stats: {summary_stats}")
            print(f"  has NaN: {torch.isnan(summary_stats).any()}")
        
        theta_samples.append(theta)
        x_samples.append(summary_stats)
    
    theta_samples = torch.stack(theta_samples)
    x_samples = torch.stack(x_samples)
    
    # Check for NaN before normalization
    num_nan = torch.isnan(x_samples).any(dim=1).sum()
    print(f"\nNumber of samples with NaN: {num_nan}/{num_simulations}")
    
    if num_nan > 0:
        print("⚠️  WARNING: NaN values detected!")
        print(f"Sample with NaN: {x_samples[torch.isnan(x_samples).any(dim=1)][0]}")
        raise ValueError("Cannot proceed with NaN values in training data")
    
    # Ensure float32
    x_samples = x_samples.float()
    
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


def train_snle(simulator, prior, num_simulations=10000, window_sites=100, 
               mode='single', max_num_epochs=50, batch_size=50):
    """
    Train SNLE to learn p(summary_stats | theta).
    
    Args:
        simulator: PatchForagingDDM instance
        prior: Prior distribution over parameters
        num_simulations: Number of training samples
        window_sites: Number of sites to simulate
        mode: 'single' for single-patch inference
              'multi' for multi-patch inference
        max_num_epochs: Maximum training epochs
        batch_size: Training batch size
    
    Returns:
        likelihood_estimator: Trained SNLE
        inference: SNLE inference object
        x_mean: Mean of training data (for normalization)
        x_std: Std of training data (for normalization)
        training_history: Dict with training history
    """
    # Generate training data
    theta_samples, x_samples, x_mean, x_std = generate_likelihood_training_data(
        simulator, prior, num_simulations, window_sites, mode=mode
    )
    
    # Initialize SNLE
    print(f"\nTraining SNLE ({mode} mode)...")
    inference = SNLE(prior=prior)
    
    # Add training data
    inference.append_simulations(theta_samples, x_samples)
    
    # Train with more detailed output
    likelihood_estimator = inference.train(
        training_batch_size=batch_size,
        max_num_epochs=max_num_epochs,
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
    
    training_history = {
        'train_loss': inference._summary['training_loss'],
        'val_loss': inference._summary['validation_loss'],
        'epochs': list(range(len(inference._summary['training_loss']))),
        'best_val_loss': best_val,
        'epochs_trained': epochs_trained,
    }
    
    print(f"\nSNLE training complete ({mode} mode)!")
    print(f"Epochs trained: {training_history['epochs_trained']}")
    print(f"Final train loss: {training_history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {training_history['val_loss'][-1]:.4f}")
    if best_val is not None:
        print(f"Best validation loss: {best_val:.4f}")
    
    return likelihood_estimator, inference, x_mean, x_std, training_history


def infer_parameters_snle(inference, observed_stats, 
                          x_mean, x_std,
                          num_samples=1000, warmup_steps=200,
                          mcmc_method="slice_np_vectorized"):
    """
    Infer parameters from observed summary statistics using MCMC.
    
    Args:
        inference: Trained SNLE inference object (contains prior)
        observed_stats: (3,) or (8,) tensor of summary statistics
        x_mean: Mean used for normalization during training
        x_std: Std used for normalization during training
        num_samples: Number of MCMC samples
        warmup_steps: MCMC warmup steps
        mcmc_method: MCMC sampling method
    
    Returns:
        posterior_samples: (num_samples, 3) parameter samples
    """
    print("Running MCMC to infer parameters...")
    
    # Normalize observed stats the same way as training data
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
    print("Testing SNLE inference module...")
    
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM, create_prior
    
    # Setup
    simulator = PatchForagingDDM(noise_std=0.0)
    prior = create_prior()
    
    # Test training (small dataset)
    print("\n1. Testing single-patch training...")
    _, inference_single, x_mean_single, x_std_single, history_single = train_snle(
        simulator, prior, num_simulations=1000, window_sites=100, 
        mode='single', max_num_epochs=10
    )
    print(f"   Single-patch model trained: {len(history_single['train_loss'])} epochs")
    
    print("\n2. Testing multi-patch training...")
    _, inference_multi, x_mean_multi, x_std_multi, history_multi = train_snle(
        simulator, prior, num_simulations=1000, window_sites=100, 
        mode='multi', max_num_epochs=10
    )
    print(f"   Multi-patch model trained: {len(history_multi['train_loss'])} epochs")
    
    # Test inference
    print("\n3. Testing inference...")
    true_theta = torch.tensor([0.6, 0.8, 0.3])
    param_gen = simulator.walk_params(true_theta)
    _, observed_stats = simulator.simulate_trial(param_gen, 100, return_aggregate=False)
    
    samples = infer_parameters_snle(
        inference_single, observed_stats,
        x_mean_single, x_std_single,
        num_samples=100, warmup_steps=50
    )
    print(f"   Posterior samples shape: {samples.shape}")
    print(f"   True theta:      {true_theta}")
    print(f"   Posterior mean:  {samples.mean(dim=0)}")
    
    # Test save/load
    print("\n4. Testing save/load...")
    model_dir = save_snle_model(inference_single, x_mean_single, x_std_single, mode='single')
    loaded_inference, loaded_mean, loaded_std, loaded_mode, analysis_dir = load_snle_model(model_dir)
    print("   Model saved and loaded successfully")
    print(f"   Analysis directory: {analysis_dir}")
    
    print("\n✓ All tests passed!")