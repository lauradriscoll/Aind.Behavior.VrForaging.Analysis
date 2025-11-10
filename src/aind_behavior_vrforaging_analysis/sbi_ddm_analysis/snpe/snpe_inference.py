"""
SBI inference for patch foraging.

Single purpose: Train neural posterior estimator and run inference.
"""

import torch
from sbi import utils as sbi_utils
from sbi.inference import SNPE
from typing import Optional

from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snpe.snpe_features import extract_features


def generate_training_data(simulator, prior, num_simulations: int, window_sites: int = 100):
    """
    Generate training data for SBI.
    
    Returns:
        theta_samples: (num_simulations, 3)
        feature_samples: (num_simulations, 611)
    """
    print(f"Generating {num_simulations} training samples...")
    
    theta_samples = []
    feature_samples = []
    
    for i in range(num_simulations):
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{num_simulations}")
        
        # Sample parameters
        theta = prior.sample()

        # Simulate window with walk around theta
        window = simulator.simulate_with_walk(
            theta_mean=theta,
            window_sites=window_sites,
            sigma=0.0
        )
        
        # Extract features
        features = extract_features(window)
        
        # Check for NaN (sbi will filter these out)
        theta_samples.append(theta)
        feature_samples.append(features)
    
    theta_samples = torch.stack(theta_samples)
    feature_samples = torch.stack(feature_samples)
    
    print(f"Training data generated: {theta_samples.shape}, {feature_samples.shape}")
    return theta_samples, feature_samples


def train_posterior(prior, theta_samples, feature_samples, batch_size: int = 50, 
                   device: str = 'cpu'):
    """
    Train neural posterior estimator.
    
    Returns:
        posterior: Trained posterior distribution
    """
    print(f"\nTraining neural network on {device}...")
    
    # Initialize
    inference = SNPE(prior=prior, device=device)
    
    # Add training data
    inference.append_simulations(theta_samples, feature_samples)
    
    # Train
    density_estimator = inference.train(
        training_batch_size=batch_size,
        max_num_epochs=50,
        show_train_summary=True
    )
    
    # Build posterior
    posterior = inference.build_posterior(density_estimator)
    
    print("Training complete!")
    return posterior


def infer_parameters(posterior, window, num_samples: int = 1000):
    """
    Infer parameters from observed window.
    
    Args:
        posterior: Trained posterior from train_posterior()
        window: (100, 3) observed behavioral data
        num_samples: Number of posterior samples
    
    Returns:
        samples: (num_samples, 3) posterior samples
    """
    features = extract_features(window)
    samples = posterior.sample((num_samples,), x=features)
    return samples


def train_sbi(simulator, prior, num_simulations: int = 50000, 
              batch_size: int = 50, device: str = 'cpu'):
    """
    Complete SBI training pipeline.
    
    Args:
        simulator: PatchForagingDDM instance
        prior: Prior over parameters
        num_simulations: Number of training samples
        batch_size: Training batch size
        device: 'cpu' or 'cuda'
    
    Returns:
        posterior: Trained posterior distribution
    """
    # Generate data
    theta_samples, feature_samples = generate_training_data(
        simulator, prior, num_simulations
    )
    
    # Train
    posterior = train_posterior(
        prior, theta_samples, feature_samples, 
        batch_size=batch_size, device=device
    )
    
    return posterior

def save_posterior(posterior, filepath: str = 'posterior.pkl'):
    """
    Save trained posterior to disk.
    
    Args:
        posterior: Trained posterior from train_sbi()
        filepath: Path to save file
    """
    torch.save(posterior, filepath)
    print(f"Posterior saved to {filepath}")


def load_posterior(filepath: str = 'posterior.pkl'):
    """
    Load trained posterior from disk.
    
    Args:
        filepath: Path to saved posterior
    
    Returns:
        posterior: Trained posterior distribution
    """
    posterior = torch.load(filepath)
    print(f"Posterior loaded from {filepath}")
    return posterior


# Test
if __name__ == "__main__":
    print("Testing inference module...")
    
    from simulator import PatchForagingDDM, create_prior
    
    # Initialize
    simulator = PatchForagingDDM()
    prior = create_prior()
    
    # Test data generation
    print("\n1. Testing data generation...")
    theta_samples, feature_samples = generate_training_data(
        simulator, prior, num_simulations=100
    )
    print(f"   Theta shape: {theta_samples.shape}")
    print(f"   Features shape: {feature_samples.shape}")
    
    # Test training (small)
    print("\n2. Testing training...")
    posterior = train_posterior(prior, theta_samples, feature_samples)
    print(f"   Posterior trained: {posterior is not None}")
    
    # Test inference
    print("\n3. Testing inference...")
    test_window = simulator.simulate_constant(torch.tensor([0.5, 0.6, 0.2]), 100)
    samples = infer_parameters(posterior, test_window, num_samples=100)
    print(f"   Posterior samples shape: {samples.shape}")
    
    print("\n✓ All tests passed!")