"""
Utilities for SNLE analysis and visualization.

Purpose: Plotting, diagnostics, and comparison functions for SNLE inference.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sbi.analysis import pairplot


def plot_training_history(training_history, mode='single', save_path=None, analysis_dir=None):
    """
    Plot training and validation loss over epochs.
    
    Args:
        training_history: Dict with 'train_loss', 'val_loss', 'epochs'
        mode: 'single' or 'multi' for plot title
        save_path: Optional path to save figure (deprecated, use analysis_dir)
        analysis_dir: If provided, saves to analysis_dir/training_history.png
    """
    import os
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    epochs = training_history['epochs']
    train_loss = training_history['train_loss']
    val_loss = training_history['val_loss']
    
    ax.plot(epochs, train_loss, label='Training Loss', linewidth=2, marker='o', markersize=3)
    ax.plot(epochs, val_loss, label='Validation Loss', linewidth=2, marker='s', markersize=3)
    
    if training_history['best_val_loss'] is not None:
        ax.axhline(training_history['best_val_loss'], color='green', linestyle='--', 
                   linewidth=1, alpha=0.7, label='Best Validation')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'SNLE Training History ({mode} mode)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Determine save path
    if analysis_dir:
        save_path = os.path.join(analysis_dir, 'training_history.png')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print diagnostics
    final_train = train_loss[-1]
    final_val = val_loss[-1]
    gap = final_val - final_train
    
    print(f"\n{'='*60}")
    print("Training Diagnostics:")
    print(f"{'='*60}")
    print(f"Epochs trained:        {training_history['epochs_trained']}")
    print(f"Final training loss:   {final_train:.4f}")
    print(f"Final validation loss: {final_val:.4f}")
    if training_history['best_val_loss'] is not None:
        print(f"Best validation loss:  {training_history['best_val_loss']:.4f}")
    print(f"Train-val gap:         {gap:.4f}")
    
    if gap > 0.5:
        print("\n⚠️  WARNING: Large train-val gap suggests overfitting!")
        print("   Consider: reducing max_num_epochs or adding regularization")
    elif gap < -0.2:
        print("\n⚠️  WARNING: Validation loss much lower than training loss (unusual)")
        print("   This might indicate issues with validation split")
    else:
        print("\n✓ Train-val gap looks reasonable")
    
    # Check if early stopping was triggered
    epochs_trained = training_history['epochs_trained']
    max_epochs = len(epochs)
    if epochs_trained < max_epochs:
        print(f"\n✓ Early stopping triggered at epoch {epochs_trained}")
    
    print(f"{'='*60}")


def plot_posterior_pairplot(prior, posterior_samples, true_theta, save_path=None):
    """
    Create pairplot comparing prior and posterior distributions.
    
    Args:
        prior: Prior distribution
        posterior_samples: (N, 3) posterior samples
        true_theta: (3,) true parameter values
        save_path: Optional path to save figure
    """
    # Generate prior samples for comparison
    prior_samples = prior.sample((1000,))
    
    # Create pairplot
    fig, ax = pairplot(
        [prior_samples, posterior_samples],
        points=true_theta,
        diag="kde",
        upper="contour", 
        kde_offdiag=dict(bins=50),
        kde_diag=dict(bins=100),
        contour_offdiag=dict(levels=[0.95]),
        points_colors=["red"], 
        points_offdiag=dict(marker="*", markersize=15), 
        labels=[r"drift_rate", r"reward_bump", r"failure_bump"],
        figsize=(10, 10)
    )
    
    # Add legend
    plt.sca(ax[2, 2])
    plt.legend(
        ["Prior", "Posterior", r"$\theta_{true}$"], 
        frameon=False, 
        fontsize=12,
        loc='upper right'
    )
    
    plt.suptitle('SNLE: Prior vs Posterior', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Pairplot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_snle_vs_simulator(simulator, true_theta, posterior_samples, 
                              num_patches=100, save_path=None):
    """
    Compare data generated from SNLE posterior vs true simulator.
    
    Args:
        simulator: PatchForagingDDM instance
        true_theta: True parameters
        posterior_samples: Posterior parameter samples
        num_patches: Number of patches to generate for comparison
        save_path: Optional path to save figure
    """
    print(f"\nGenerating {num_patches} patches from each model...")
    
    # 1. Generate "real" data from true simulator
    real_patches = []
    for _ in range(num_patches):
        global_time = 0.0
        patch_start_time = 0.0
        _, _, stats = simulator._simulate_one_patch(true_theta, global_time, patch_start_time)
        real_patches.append(stats)
    real_data = torch.stack(real_patches)
    
    # 2. Generate "synthetic" data from posterior-sampled parameters
    synthetic_patches = []
    for i in range(num_patches):
        theta_sample = posterior_samples[i % len(posterior_samples)]
        global_time = 0.0
        patch_start_time = 0.0
        _, _, stats = simulator._simulate_one_patch(theta_sample, global_time, patch_start_time)
        synthetic_patches.append(stats)
    synthetic_data = torch.stack(synthetic_patches)
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Time in patch distributions
    axes[0].hist(synthetic_data[:, 0].numpy(), bins=30, alpha=0.7, label='SNLE', 
                 density=True, edgecolor='black')
    axes[0].hist(real_data[:, 0].numpy(), bins=30, alpha=0.7, label='Simulator', 
                 density=True, edgecolor='black')
    axes[0].set_xlabel('Total Time in Patch')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Patch Duration Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Average rewards comparison
    synth_rewards = synthetic_data[:, 2].mean()
    real_rewards = real_data[:, 2].mean()
    
    axes[1].bar(['SNLE', 'Simulator'], [synth_rewards, real_rewards], 
                alpha=0.7, edgecolor='black', capsize=5)
    axes[1].set_ylabel('Average Rewards per Patch')
    axes[1].set_title('Mean Rewards Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, (label, value) in enumerate(zip(['SNLE', 'Simulator'], [synth_rewards, real_rewards])):
        axes[1].text(i, value + 0.1, f'{value:.2f}', ha='center', fontsize=10)
    
    # Plot 3: Rewards vs Time scatter
    axes[2].scatter(synthetic_data[:, 2].numpy() + np.random.normal(0, 0.05, num_patches), 
                    synthetic_data[:, 0].numpy(), alpha=0.4, label='SNLE', s=20)
    axes[2].scatter(real_data[:, 2].numpy() + np.random.normal(0, 0.05, num_patches), 
                    real_data[:, 0].numpy(), alpha=0.4, label='Simulator', s=20)
    axes[2].set_xlabel('Number of Rewards (jittered)')
    axes[2].set_ylabel('Total Time in Patch')
    axes[2].set_title('Rewards vs Time Relationship')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('SNLE vs Simulator: Patch Statistics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"\nTotal Time in Patch:")
    print(f"  SNLE:      {synthetic_data[:, 0].mean():.3f} ± {synthetic_data[:, 0].std():.3f}")
    print(f"  Simulator: {real_data[:, 0].mean():.3f} ± {real_data[:, 0].std():.3f}")
    print(f"\nNumber of Stops:")
    print(f"  SNLE:      {synthetic_data[:, 1].mean():.3f} ± {synthetic_data[:, 1].std():.3f}")
    print(f"  Simulator: {real_data[:, 1].mean():.3f} ± {real_data[:, 1].std():.3f}")
    print(f"\nNumber of Rewards:")
    print(f"  SNLE:      {synthetic_data[:, 2].mean():.3f} ± {synthetic_data[:, 2].std():.3f}")
    print(f"  Simulator: {real_data[:, 2].mean():.3f} ± {real_data[:, 2].std():.3f}")
    print(f"{'='*60}")


def plot_posterior_distributions(posterior_samples, true_theta=None, save_path=None):
    """
    Plot marginal posterior distributions for each parameter.
    
    Args:
        posterior_samples: (N, 3) posterior samples
        true_theta: Optional true parameter values
        save_path: Optional path to save figure
    """
    samples_np = posterior_samples.cpu().numpy()
    param_names = ['drift_rate', 'reward_bump', 'failure_bump']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(samples_np[:, i], bins=50, density=True, alpha=0.7, 
               edgecolor='black', color='blue')
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.set_title(f'Posterior: {name}')
        
        if true_theta is not None:
            ax.axvline(true_theta[i].item(), color='red', linestyle='--', 
                      linewidth=2, label='True value')
            ax.legend()
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Posterior distributions saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_inference_summary(posterior_samples, true_theta):
    """
    Print summary statistics of inference results.
    
    Args:
        posterior_samples: (N, 3) posterior samples
        true_theta: (3,) true parameter values
    """
    posterior_mean = posterior_samples.mean(dim=0)
    posterior_std = posterior_samples.std(dim=0)
    absolute_error = (posterior_mean - true_theta).abs()
    relative_error = absolute_error / (true_theta.abs() + 1e-6)
    
    param_names = ['drift_rate', 'reward_bump', 'failure_bump']
    
    print(f"\n{'='*60}")
    print("INFERENCE RESULTS")
    print(f"{'='*60}")
    print(f"\n{'Parameter':15s} {'True':>10s} {'Mean':>10s} {'Std':>10s} {'AbsErr':>10s} {'RelErr':>10s}")
    print("-" * 60)
    
    for i, name in enumerate(param_names):
        print(f"{name:15s} {true_theta[i].item():10.4f} "
              f"{posterior_mean[i].item():10.4f} {posterior_std[i].item():10.4f} "
              f"{absolute_error[i].item():10.4f} {relative_error[i].item():10.2%}")
    
    print(f"{'='*60}")
    
    # Overall assessment
    mean_abs_error = absolute_error.mean().item()
    mean_rel_error = relative_error.mean().item()
    
    print(f"\nOverall Performance:")
    print(f"  Mean absolute error: {mean_abs_error:.4f}")
    print(f"  Mean relative error: {mean_rel_error:.2%}")
    
    if mean_abs_error < 0.1:
        print("\n✓ Excellent parameter recovery!")
    elif mean_abs_error < 0.3:
        print("\n✓ Good parameter recovery")
    else:
        print("\n⚠️  Poor parameter recovery - consider more training data or better features")
    
    print(f"{'='*60}\n")

def save_snle_model(inference, x_mean, x_std, mode='single', base_dir='snle_models'):
    """
    Save trained SNLE model and normalization parameters in timestamped folder.
    
    Args:
        inference: Trained SNLE inference object
        x_mean: Training data mean
        x_std: Training data std
        mode: 'single' or 'multi' for folder naming
        base_dir: Base directory for all models
    
    Returns:
        model_dir: Path to the created model directory
    """
    from datetime import datetime
    import os
    
    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(base_dir, f'{mode}_patch_{timestamp}')
    analysis_dir = os.path.join(model_dir, 'analysis')
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Save model
    model_dict = {
        'inference': inference,
        'x_mean': x_mean,
        'x_std': x_std,
        'mode': mode,
        'timestamp': timestamp,
    }
    model_path = os.path.join(model_dir, 'model.pkl')
    torch.save(model_dict, model_path)
    
    print(f"SNLE model saved to: {model_dir}")
    print(f"  - Model: {model_path}")
    print(f"  - Analysis folder: {analysis_dir}")
    
    return model_dir


def load_snle_model(model_dir):
    """
    Load trained SNLE model and normalization parameters from timestamped folder.
    
    Args:
        model_dir: Path to model directory (e.g., 'snle_models/multi_patch_20241107_143022')
    
    Returns:
        inference: Trained SNLE inference object
        x_mean: Training data mean
        x_std: Training data std
        mode: 'single' or 'multi'
        analysis_dir: Path to analysis folder
    """
    import os
    
    model_path = os.path.join(model_dir, 'model.pkl')
    analysis_dir = os.path.join(model_dir, 'analysis')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model_dict = torch.load(model_path)
    
    print(f"SNLE model loaded from: {model_dir}")
    print(f"  - Mode: {model_dict['mode']}")
    print(f"  - Timestamp: {model_dict['timestamp']}")
    print(f"  - Analysis folder: {analysis_dir}")
    
    return (model_dict['inference'], model_dict['x_mean'], model_dict['x_std'], 
            model_dict['mode'], analysis_dir)


def list_saved_models(base_dir='snle_models'):
    """
    List all saved SNLE models.
    
    Args:
        base_dir: Base directory for all models
    
    Returns:
        models: List of (model_dir, mode, timestamp) tuples
    """
    import os
    from datetime import datetime
    
    if not os.path.exists(base_dir):
        print(f"No models found (directory {base_dir} does not exist)")
        return []
    
    models = []
    for folder in sorted(os.listdir(base_dir), reverse=True):
        model_dir = os.path.join(base_dir, folder)
        if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, 'model.pkl')):
            # Parse folder name: mode_patch_timestamp
            parts = folder.split('_')
            if len(parts) >= 3:
                mode = parts[0]
                timestamp = '_'.join(parts[2:])
                models.append((model_dir, mode, timestamp))
    
    print(f"\nFound {len(models)} saved models:")
    for i, (model_dir, mode, timestamp) in enumerate(models):
        print(f"  {i+1}. {mode:6s} - {timestamp} - {model_dir}")
    
    return models


# Test
if __name__ == "__main__":
    print("Testing SNLE utilities...")
    
    # Create dummy data
    training_history = {
        'train_loss': [2.5, 2.0, 1.8, 1.6, 1.5],
        'val_loss': [2.6, 2.1, 1.9, 1.7, 1.6],
        'epochs': [0, 1, 2, 3, 4],
        'best_val_loss': 1.6,
        'epochs_trained': 5,
    }
    
    true_theta = torch.tensor([0.6, 0.8, 0.3])
    posterior_samples = torch.randn(1000, 3) * 0.1 + true_theta
    
    print("\n1. Testing training history plot...")
    plot_training_history(training_history, mode='single')
    
    print("\n2. Testing posterior distributions plot...")
    plot_posterior_distributions(posterior_samples, true_theta)
    
    print("\n3. Testing inference summary...")
    print_inference_summary(posterior_samples, true_theta)
    
    print("\n✓ All utility tests passed!")