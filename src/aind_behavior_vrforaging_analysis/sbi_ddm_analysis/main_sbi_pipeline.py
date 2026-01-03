"""
Main script to run full SBI pipeline for VR foraging DDM parameter inference

Workflow:
1. Train SNLE model (once, 2M simulations)
2. Validate parameter recovery on simulated data
3. Load real data for each odor type
4. Infer parameters per window for each odor
5. Analyze temporal dynamics
6. Compare across odors
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'False'  # Disable 64-bit (faster)

import numpy as np
import jax.numpy as jnp
from jax import random
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM_JAX, create_prior
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.sbi_pipeline import (
    train_snle_model,
    validate_parameter_recovery,
    infer_parameters_per_window,
    analyze_temporal_dynamics,
    compare_odors_temporal
)

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Data paths
    'base_path': Path("/Users/laura.driscoll/Documents/data/VR foraging/vr_foraging_data"),
    'output_dir': Path("/Users/laura.driscoll/Documents/code/sbi_results"),
    
    'odor_types': ['Methyl_Butyrate', 'Alpha_pinene'],
    'odor_display_names': {'Methyl_Butyrate': 'Methyl Butyrate', 'Alpha_pinene': 'Alpha-pinene'},
    
    # Simulator parameters
    'interval_min': 20.0,
    'interval_scale': 19.0,
    'odor_site_length': 50.0,
    'interval_normalization': 88.73,
    
    # Prior ranges
    'prior_low': [0.0, 0.0, 0.0, 0.05],
    'prior_high': [2.0, 2.0, 2.0, 0.5],
    
    # Training settings - 2M samples
    'n_simulations': 10_000,
    'n_iter': 500,
    'batch_size': 128,
    'n_early_stopping_patience': 50,
    'learning_rate': 1e-3,
    'hidden_dim': 64,
    'num_layers': 5,

    # Window settings
    'window_size': 100,  # Number of sites per window
    'step_size': 10,     # Overlap between windows (used during extraction)
    
    # Validation settings
    'n_validation_tests': 100,
    
    # Inference settings (per window)
    'num_samples': 500,
    'num_warmup': 200,
    'num_chains': 2,
    
    # Engagement filter
    'apply_engagement_filter': True,
    'min_entry_rate': 0.80,
    
    'seed': 42
}

CONFIG['output_dir'].mkdir(exist_ok=True, parents=True)

param_names = ['drift_rate', 'reward_bump', 'failure_bump', 'noise_std']

print(f"\n{'='*70}")
print(f"VR FORAGING DDM PARAMETER INFERENCE")
print(f"{'='*70}")
print(f"\nConfiguration:")
print(f"  Training with {CONFIG['n_simulations']:,} simulations")
print(f"  Network: {CONFIG['num_layers']} layers, {CONFIG['hidden_dim']} hidden units")
print(f"  Prior ranges:")
for i, name in enumerate(param_names):
    print(f"    {name:15s}: [{CONFIG['prior_low'][i]:.2f}, {CONFIG['prior_high'][i]:.2f}]")

# ============================================================================
# Initialize simulator and prior
# ============================================================================

print(f"\n{'='*70}")
print(f"INITIALIZING SIMULATOR AND PRIOR")
print(f"{'='*70}")

simulator = PatchForagingDDM_JAX(
    initial_prob=0.8,
    decay_rate=-0.1,
    threshold=1.0,
    start_point=0.0,
    interval_min=CONFIG['interval_min'],
    interval_scale=CONFIG['interval_scale'],
    interval_normalization=CONFIG['interval_normalization'],
    odor_site_length=CONFIG['odor_site_length'],
    max_sites_per_window=CONFIG['window_size']
)

prior_fn = create_prior(
    prior_low=jnp.array(CONFIG['prior_low']),
    prior_high=jnp.array(CONFIG['prior_high'])
)

print("Simulator and prior initialized")

# ============================================================================
# STEP 1: Train SNLE model
# ============================================================================

model_path = CONFIG['output_dir'] / "snle_model_2M.pkl"

if model_path.exists():
    print(f"\n{'='*70}")
    print(f"Loading existing trained model from: {model_path}")
    print(f"{'='*70}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print("Model loaded successfully")
else:
    print(f"\n{'='*70}")
    print(f"STEP 1: TRAINING SNLE MODEL")
    print(f"{'='*70}")
    print(f"This will take 30 min - 2 hours with {CONFIG['n_simulations']:,} simulations...")
    
    model_data = train_snle_model(
        simulator,
        prior_fn,
        CONFIG,
        save_path=model_path
    )
    
    # Plot training loss
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(model_data['losses']['train_losses'], label='Train', linewidth=2)
    ax.plot(model_data['losses']['val_losses'], label='Validation', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('SNLE Training Loss (2M samples)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    loss_plot_path = CONFIG['output_dir'] / "training_loss_2M.png"
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training loss plot saved to: {loss_plot_path}")

# ============================================================================
# STEP 2: Validate parameter recovery
# ============================================================================

validation_path = CONFIG['output_dir'] / "validation_results.npz"

if validation_path.exists():
    print(f"\n{'='*70}")
    print(f"Loading existing validation results from: {validation_path}")
    print(f"{'='*70}")
    
    val_data = np.load(validation_path)
    validation_results = {
        'true_params': val_data['true_params'],
        'inferred_params': val_data['inferred_params'],
        'param_names': param_names
    }
    
    # Recreate figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, name in enumerate(param_names):
        ax = axes[i]
        ax.scatter(validation_results['true_params'][:, i], 
                  validation_results['inferred_params'][:, i], 
                  alpha=0.5, s=30)
        
        min_val = min(validation_results['true_params'][:, i].min(), 
                     validation_results['inferred_params'][:, i].min())
        max_val = max(validation_results['true_params'][:, i].max(), 
                     validation_results['inferred_params'][:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect recovery')
        
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Inferred {name}')
        ax.set_title(f'{name} Recovery')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
else:
    print(f"\n{'='*70}")
    print(f"STEP 2: VALIDATING PARAMETER RECOVERY")
    print(f"{'='*70}")
    
    validation_results, fig = validate_parameter_recovery(
        model_data,
        simulator,
        prior_fn,
        n_test=CONFIG['n_validation_tests'],
        seed=CONFIG['seed']
    )
    
    # Save validation results
    np.savez(
        validation_path,
        true_params=validation_results['true_params'],
        inferred_params=validation_results['inferred_params']
    )
    print(f"Validation results saved to: {validation_path}")

# Save validation plot
val_plot_path = CONFIG['output_dir'] / "parameter_recovery.png"
plt.savefig(val_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Parameter recovery plot saved to: {val_plot_path}")

# ============================================================================
# STEP 3: Load real data for each odor type
# ============================================================================

print(f"\n{'='*70}")
print(f"STEP 3: LOADING REAL DATA")
print(f"{'='*70}")

def load_windows_for_odor(base_path, odor_type, engagement_filter=True, min_entry_rate=0.80):
    """Load all windows for a specific odor type, optionally filtering for engagement."""
    all_windows = []
    
    results_df = pd.read_csv(base_path / "batch_processing_by_odor_results.csv")
    successful_sessions = results_df[results_df['status'] == 'success']
    
    for idx, row in successful_sessions.iterrows():
        session_dir = Path(row['session_dir'])
        # Use window size in folder name
        odor_dir = session_dir / f"{window_size}_window_data_by_odor" / odor_type
        
        if not odor_dir.exists():
            continue
        
        import json
        with open(odor_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        for i in range(metadata['n_windows']):
            window = np.load(odor_dir / f"window_{i:03d}.npy")
            all_windows.append(window)
    
    windows = np.array(all_windows)
    
    if not engagement_filter:
        return windows
    
    # Apply engagement filter
    engaged_windows = []
    
    for w in range(windows.shape[0]):
        positions = windows[w, :, 0]
        stopped = windows[w, :, 2]
        
        n_patches = 0
        n_entered = 0
        current_patch_max_pos = 0
        
        for i in range(len(stopped)):
            if stopped[i] == 1:
                current_patch_max_pos = max(current_patch_max_pos, positions[i])
            else:
                n_patches += 1
                if current_patch_max_pos > 1.0:
                    n_entered += 1
                current_patch_max_pos = 0
        
        if n_patches > 0:
            entry_rate = n_entered / n_patches
            if entry_rate >= min_entry_rate:
                engaged_windows.append(windows[w])
    
    return np.array(engaged_windows) if engaged_windows else np.array([]).reshape(0, windows.shape[1], windows.shape[2])

observed_data = {}

for odor_type in CONFIG['odor_types']:
    odor_name = CONFIG['odor_display_names'][odor_type]
    
    print(f"\nLoading {odor_name}...")
    
    windows = load_windows_for_odor(
        CONFIG['base_path'],
        odor_type,
        engagement_filter=CONFIG['apply_engagement_filter'],
        min_entry_rate=CONFIG['min_entry_rate']
    )
    
    print(f"  Loaded {len(windows)} windows")
    
    observed_data[odor_type] = {
        'windows': windows,
        'odor_name': odor_name
    }

# ============================================================================
# STEP 4: Infer parameters per window for each odor
# ============================================================================

print(f"\n{'='*70}")
print(f"STEP 4: INFERRING PARAMETERS PER WINDOW")
print(f"{'='*70}")

posterior_results = {}

for odor_type in CONFIG['odor_types']:
    odor_name = observed_data[odor_type]['odor_name']
    windows = observed_data[odor_type]['windows']
    
    print(f"\n{'-'*70}")
    print(f"Processing {odor_name}")
    print(f"{'-'*70}")
    
    # Check if already processed
    posterior_path = CONFIG['output_dir'] / f"posterior_per_window_{odor_type}.pkl"
    
    if posterior_path.exists():
        print(f"Loading existing results from: {posterior_path}")
        with open(posterior_path, 'rb') as f:
            posterior_samples_per_window = pickle.load(f)
    else:
        posterior_samples_per_window = infer_parameters_per_window(
            model_data,
            windows,
            seed=CONFIG['seed'] + hash(odor_type) % 1000  # Different seed per odor
        )
        
        # Save results
        with open(posterior_path, 'wb') as f:
            pickle.dump(posterior_samples_per_window, f)
        print(f"Saved posterior samples to: {posterior_path}")
    
    posterior_results[odor_type] = posterior_samples_per_window

# ============================================================================
# STEP 5: Analyze temporal dynamics for each odor
# ============================================================================

print(f"\n{'='*70}")
print(f"STEP 5: ANALYZING TEMPORAL DYNAMICS")
print(f"{'='*70}")

temporal_results = {}

for odor_type in CONFIG['odor_types']:
    odor_name = observed_data[odor_type]['odor_name']
    
    print(f"\n{'-'*70}")
    print(f"Analyzing {odor_name}")
    print(f"{'-'*70}")
    
    summary_stats, fig = analyze_temporal_dynamics(
        posterior_results[odor_type],
        param_names,
        odor_name=odor_name,
        save_path=CONFIG['output_dir'] / f"temporal_dynamics_{odor_type}.png"
    )
    
    temporal_results[odor_type] = summary_stats
    plt.close(fig)

# ============================================================================
# STEP 6: Compare across odors
# ============================================================================

print(f"\n{'='*70}")
print(f"STEP 6: COMPARING ACROSS ODORS")
print(f"{'='*70}")

fig = compare_odors_temporal(
    temporal_results,
    param_names,
    save_path=CONFIG['output_dir'] / "odor_comparison_temporal.png"
)

plt.show()

# ============================================================================
# Summary
# ============================================================================

print(f"\n{'='*70}")
print(f"PIPELINE COMPLETE!")
print(f"{'='*70}")
print(f"\nResults saved to: {CONFIG['output_dir']}")
print(f"\nGenerated files:")
print(f"  - snle_model_2M.pkl: Trained model")
print(f"  - training_loss_2M.png: Training curves")
print(f"  - parameter_recovery.png: Validation results")
print(f"  - posterior_per_window_*.pkl: Inferred parameters per window")
print(f"  - temporal_dynamics_*.png: Parameter evolution over session")
print(f"  - odor_comparison_temporal.png: Cross-odor comparison")

print(f"\n{'='*70}")