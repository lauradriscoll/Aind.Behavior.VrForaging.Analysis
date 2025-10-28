"""
Train an MNLE (Mixed Neural Likelihood Estimator) for DDM parameter inference.

Usage:
    python train_mnle.py --n_simulations 50000 --output_dir models/
"""

import sys
sys.path.append('../src')

import argparse
import os
import pickle
import json
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sbi.neural_nets import likelihood_nn

from sbi.inference import MNLE
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import DDMSimulator, create_ddm_prior


def setup_ddm():
    """
    Initialize DDM simulator and prior.
    
    Returns:
        simulator: DDMSimulator instance
        prior: Prior distribution for [drift, reward_pulse]
    """
    print("Setting up DDM simulator and prior...")
    simulator = DDMSimulator()
    prior = create_ddm_prior()
    
    print("DDM Parameters:")
    print("  - drift: evidence accumulation rate")
    print("  - reward_pulse: size of reward bumps")
    
    return simulator, prior


def compute_max_rewards(simulator, prior, n_batches=10, batch_size=1000):
    """
    Compute maximum reward count across multiple batches to set safe bound.
    
    Args:
        simulator: DDMSimulator instance
        prior: Prior distribution
        n_batches: Number of batches to sample
        batch_size: Size of each batch
        
    Returns:
        max_rewards: Safe upper bound for reward counts (rounded up)
    """
    print(f"\nComputing MAX_REWARDS from {n_batches} batches of {batch_size} samples...")
    max_rewards_list = []
    
    for i in range(n_batches):
        theta_batch = prior.sample((batch_size,))
        x_batch = torch.stack([simulator(theta_i) for theta_i in theta_batch])
        max_rewards_list.append(x_batch[:, 1].max().item())
        
    max_rewards = int(np.ceil(max(max_rewards_list)))
    print(f"MAX_REWARDS set to: {max_rewards}")
    print(f"Range across batches: [{min(max_rewards_list):.1f}, {max(max_rewards_list):.1f}]")
    
    return max_rewards


def generate_training_data(simulator, prior, n_simulations, seed=42):
    """
    Generate training data for MNLE.
    
    Args:
        simulator: DDMSimulator instance
        prior: Prior distribution
        n_simulations: Number of simulations to generate
        seed: Random seed for reproducibility
        
    Returns:
        theta: Parameter samples [n_simulations, n_params]
        x: Simulation outputs [n_simulations, n_outputs]
    """
    print(f"\nGenerating {n_simulations} training simulations...")
    torch.manual_seed(seed)
    
    # Sample parameters
    theta = prior.sample((n_simulations,))
    
    # Run simulations
    x = torch.stack([simulator(theta[i]) for i in range(n_simulations)])
    
    print(f"Training data shapes: theta={theta.shape}, x={x.shape}")
    print(f"Time range: [{x[:, 0].min():.2f}, {x[:, 0].max():.2f}]")
    print(f"Rewards range: [{x[:, 1].min():.0f}, {x[:, 1].max():.0f}]")
    
    return theta, x

def train_mnle_model(theta, x, prior):
    """
    Train MNLE model.
    
    Args:
        theta: Parameter samples
        x: Simulation outputs
        
    Returns:
        estimator: Trained likelihood estimator
        trainer: MNLE trainer object (for building posterior)
    """
    print("\nTraining MNLE...")
    
    # Validate that training data covers the expected range
    unique_rewards = torch.unique(x[:, 1])
    print(f"unique_rewards: {unique_rewards}")
    print(f"len(unique_rewards): {len(unique_rewards)}")
    print(f"num_categories being passed: {len(unique_rewards)+1}")
    print(f"Actual range in data: {x[:, -1].min()} to {x[:, -1].max()}")
    print(f"Unique values in x: {np.unique(x[:, -1])}")

    # Before training, remap rewards to consecutive indices
    unique_rewards_sorted = torch.sort(torch.unique(x[:, -1]))[0]
    reward_to_index = {reward.item(): idx for idx, reward in enumerate(unique_rewards_sorted)}

    # Remap the categorical column(s) in x
    x_remapped = x.clone()
    x_remapped[:, -1] = torch.tensor([reward_to_index[val.item()] for val in x[:, -1]])

    # Now you have consecutive categories 0-20
    num_categories = len(unique_rewards_sorted)  # = 21
    estimator_builder = likelihood_nn(model="mnle", log_transform_x=True, num_categories=num_categories)
    
    # Train MNLE and obtain MCMC-based posterior.
    proposal=prior
    trainer = MNLE(proposal, estimator_builder)
    estimator = trainer.append_simulations(theta, x_remapped).train()
    print("Training completed!")
    
    return estimator, trainer


def validate_emulator(estimator, simulator, prior, test_theta, n_samples=1000):
    """
    Validate MNLE emulator against true simulator.
    
    Args:
        estimator: Trained likelihood estimator
        simulator: True DDM simulator
        prior: Prior distribution
        test_theta: Test parameters (or None to sample from prior)
        n_samples: Number of samples for comparison
        
    Returns:
        validation_results: Dict with synthetic and real data
    """
    print("\nValidating MNLE emulator...")
    
    # Sample test parameters if not provided
    if test_theta is None:
        test_theta = prior.sample((1,))
    
    print(f"Test parameters: drift={test_theta[0, 0]:.3f}, reward_pulse={test_theta[0, 1]:.3f}")
    
    # Generate synthetic data from MNLE
    synthetic_data = estimator.sample(
        sample_shape=torch.Size([n_samples]), 
        condition=test_theta
    ).squeeze(1)
    
    # Generate real data from simulator
    real_data = torch.stack([simulator(test_theta[0]) for _ in range(n_samples)])
    
    # Compute statistics
    results = {
        'test_theta': test_theta.numpy(),
        'synthetic': {
            'time_mean': synthetic_data[:, 0].mean().item(),
            'time_std': synthetic_data[:, 0].std().item(),
            'rewards_mean': synthetic_data[:, 1].mean().item(),
            'rewards_std': synthetic_data[:, 1].std().item(),
        },
        'real': {
            'time_mean': real_data[:, 0].mean().item(),
            'time_std': real_data[:, 0].std().item(),
            'rewards_mean': real_data[:, 1].mean().item(),
            'rewards_std': real_data[:, 1].std().item(),
        }
    }
    
    print("\nValidation Results:")
    print(f"  Time - MNLE: {results['synthetic']['time_mean']:.3f} ± {results['synthetic']['time_std']:.3f}")
    print(f"  Time - Sim:  {results['real']['time_mean']:.3f} ± {results['real']['time_std']:.3f}")
    print(f"  Rewards - MNLE: {results['synthetic']['rewards_mean']:.2f} ± {results['synthetic']['rewards_std']:.2f}")
    print(f"  Rewards - Sim:  {results['real']['rewards_mean']:.2f} ± {results['real']['rewards_std']:.2f}")
    
    return results, synthetic_data, real_data


def plot_validation(synthetic_data, real_data, output_dir):
    """
    Create validation plots comparing MNLE to simulator.
    
    Args:
        synthetic_data: Samples from MNLE
        real_data: Samples from simulator
        output_dir: Directory to save plots
    """
    print(f"\nCreating validation plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Time in patch distributions
    axes[0].hist(synthetic_data[:, 0].numpy(), bins=30, alpha=0.7, 
                 label='MNLE', density=True, color='blue')
    axes[0].hist(real_data[:, 0].numpy(), bins=30, alpha=0.7, 
                 label='Simulator', density=True, color='orange')
    axes[0].set_xlabel('Time in patch')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Time Distribution')
    axes[0].legend()
    
    # Rewards comparison
    synth_mean = synthetic_data[:, 1].mean()
    real_mean = real_data[:, 1].mean()
    axes[1].bar(['MNLE', 'Simulator'], [synth_mean, real_mean], 
                alpha=0.7, color=['blue', 'orange'])
    axes[1].set_ylabel('Average Rewards')
    axes[1].set_title('Mean Rewards')
    
    # Scatter plot
    axes[2].scatter(synthetic_data[:, 1].numpy() + np.random.normal(0, 0.1, len(synthetic_data)), 
                    synthetic_data[:, 0].numpy(), alpha=0.1, label='MNLE', s=10, color='blue')
    axes[2].scatter(real_data[:, 1].numpy() + np.random.normal(0, 0.1, len(real_data)), 
                    real_data[:, 0].numpy(), alpha=0.1, label='Simulator', s=10, color='orange')
    axes[2].set_xlabel('# Rewards (jittered)')
    axes[2].set_ylabel('Time in patch')
    axes[2].set_title('Rewards vs Time')
    axes[2].legend()
    
    plt.tight_layout()
    
    # Save
    plot_path = os.path.join(output_dir, 'validation_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved validation plot to: {plot_path}")
    plt.close()


def save_model(estimator, trainer, prior, metadata, output_dir):
    """
    Save trained MNLE model and metadata.
    
    Args:
        estimator: Trained likelihood estimator
        trainer: MNLE trainer object
        prior: Prior distribution
        metadata: Dict with training info
        output_dir: Directory to save model
    """
    print(f"\nSaving model to: {output_dir}")
    
    # Save estimator and trainer
    model_path = os.path.join(output_dir, 'mnle_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'estimator': estimator,
            'trainer': trainer,
            'prior': prior,
        }, f)
    print(f"  Model saved: {model_path}")
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        """Recursively convert numpy arrays to lists."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    metadata_serializable = convert_to_serializable(metadata)
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata_serializable, f, indent=2)
    print(f"  Metadata saved: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train MNLE for DDM parameter inference'
    )
    parser.add_argument('--n_simulations', type=int, default=50000,
                        help='Number of training simulations')
    parser.add_argument('--output_dir', type=str, default='models/',
                        help='Base directory for saving models')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--n_validation', type=int, default=1000,
                        help='Number of samples for validation')
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    print("="*60)
    print("MNLE Training Pipeline")
    print("="*60)
    print(f"Output directory: {run_dir}")
    print(f"Training simulations: {args.n_simulations}")
    print(f"Random seed: {args.seed}")
    
    # Setup
    simulator, prior = setup_ddm()

    # Generate training data
    theta, x = generate_training_data(
        simulator, prior, args.n_simulations, seed=args.seed
    )
    
    # Train model
    estimator, trainer = train_mnle_model(theta, x, prior)

    # Validate
    val_results, synthetic_data, real_data = validate_emulator(
        estimator, simulator, prior, test_theta=None, n_samples=args.n_validation
    )
    
    # Plot validation
    plot_validation(synthetic_data, real_data, run_dir)
    
    # Save model and metadata
    metadata = {
        'n_simulations': args.n_simulations,
        'seed': args.seed,
        'timestamp': timestamp,
        'validation_results': val_results,
    }
    save_model(estimator, trainer, prior, metadata, run_dir)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Model saved to: {run_dir}")
    print("="*60)


if __name__ == '__main__':
    main()