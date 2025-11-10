"""
Integration guide: Adding sliding window analysis to your SNLE workflow
"""

# ==============================================================================
# SCENARIO 1: You already have a trained SNLE model
# ==============================================================================

from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_utils import load_snle_model
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_sliding_window import (
    sliding_window_inference, plot_parameter_evolution
)

# Load your existing model
model_dir = "/path/to/your/snle_model_20250108_123456"
inference, x_mean, x_std, mode, analysis_dir = load_snle_model(model_dir)

# Make sure it's multi-patch mode
assert mode == 'multi', "Sliding window requires multi-patch SNLE model"

# Load your real data (or simulated data)
# Format: (num_sites, 3) tensor with [time, stops, rewards] per site
import torch
session_data = torch.load('your_session_data.pt')  # Replace with your data loading

# Run sliding window analysis
simulator = PatchForagingDDM()
results = sliding_window_inference(
    simulator, session_data,
    inference, x_mean, x_std,
    window_size=100,
    stride=25,
    num_samples=500
)

# Visualize
plot_parameter_evolution(results, save_path='my_session_analysis.png')


# ==============================================================================
# SCENARIO 2: Training a new model specifically for sliding window
# ==============================================================================

from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM, create_prior
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_inference import train_snle
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_utils import save_snle_model

# Setup
simulator = PatchForagingDDM()
prior = create_prior()

# Train with optimal hyperparameters from your parameter sweep
_, inference, x_mean, x_std, history = train_snle(
    simulator, prior,
    num_simulations=50000,  # Use value from your parameter sweep
    window_sites=100,       # Use value from your parameter sweep
    mode='multi',           # MUST be 'multi' for sliding window
    max_num_epochs=50
)

# Save model
model_dir = save_snle_model(inference, x_mean, x_std, mode='multi')
print(f"Model saved to: {model_dir}")

# Now use for sliding window analysis (see Scenario 1)


# ==============================================================================
# SCENARIO 3: Batch processing multiple sessions
# ==============================================================================

from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_sliding_window import analyze_dataset
import os

# Assuming you have a dataset structure (simulated or real)
# Format: dataset['data'][mouse_id][session_id] = (session_data, true_params)

# If you have real data, create this structure:
dataset = {
    'mice': [0, 1, 2, 3, 4],
    'data': {},
    'evolution_types': {0: 'unknown', 1: 'unknown', 2: 'unknown', 3: 'unknown', 4: 'unknown'},
    'base_params': {}
}

# Load all your sessions
for mouse_id in range(5):
    dataset['data'][mouse_id] = {}
    for session_id in range(10):
        session_data = load_real_session(mouse_id, session_id)  # Your loading function
        dataset['data'][mouse_id][session_id] = (session_data, None)  # No true params for real data

# Batch analyze
analysis_dir = 'real_data_sliding_window_analysis'
os.makedirs(analysis_dir, exist_ok=True)

all_results = analyze_dataset(
    simulator, dataset,
    inference, x_mean, x_std,
    window_size=100,
    stride=25,
    num_samples=500,
    analysis_dir=analysis_dir
)

# All plots will be saved to analysis_dir


# ==============================================================================
# SCENARIO 4: Testing on simulated data first
# ==============================================================================

from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_sliding_window import (
    simulate_full_dataset, analyze_dataset
)

# Simulate realistic data matching your experiment structure
dataset = simulate_full_dataset(
    simulator,
    num_mice=5,
    num_sessions=10,
    sites_per_session=(350, 400),
    base_params=torch.tensor([0.2, 0.5, 0.2]),
    param_std=0.05,
    drift_increase=0.3
)

# Test your pipeline on simulated data
all_results = analyze_dataset(
    simulator, dataset,
    inference, x_mean, x_std,
    window_size=100,
    stride=25,
    num_samples=500,
    analysis_dir='test_on_simulated_data'
)

# Check if recovered parameters match true parameters
# This validates your pipeline before using real data


# ==============================================================================
# SCENARIO 5: Custom stride and window size experiments
# ==============================================================================

# Test different configurations to find optimal settings
import pandas as pd

test_configs = [
    {'window_size': 100, 'stride': 10},  # High temporal resolution
    {'window_size': 100, 'stride': 25},  # Balanced
    {'window_size': 100, 'stride': 50},  # Fast
    {'window_size': 150, 'stride': 25},  # Larger window
]

# Test on one session
mouse_id = 0
session_id = 0
session_data, true_params = dataset['data'][mouse_id][session_id]

comparison_results = []

for config in test_configs:
    results = sliding_window_inference(
        simulator, session_data,
        inference, x_mean, x_std,
        window_size=config['window_size'],
        stride=config['stride'],
        num_samples=200  # Fewer samples for quick comparison
    )
    
    # Compute recovery error if true params available
    if true_params is not None:
        # Sample true params at window centers
        true_drift_at_windows = true_params[results['window_centers'].long(), 0]
        inferred_drift = results['posterior_means'][:, 0]
        
        mae = (true_drift_at_windows - inferred_drift).abs().mean().item()
        
        comparison_results.append({
            'window_size': config['window_size'],
            'stride': config['stride'],
            'num_windows': results['num_windows'],
            'mae_drift': mae,
        })

# Print comparison
df = pd.DataFrame(comparison_results)
print(df)
# Choose configuration with best MAE vs computation trade-off


# ==============================================================================
# SCENARIO 6: Working with your existing experiment scripts
# ==============================================================================

# After running snle_experiments.py to train a model:
# python snle_experiments.py full multi

# Load that model for sliding window analysis
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_experiments import experiment_multi_patch_training
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_sliding_window_experiments import experiment_sliding_window_pipeline

# Option A: Train fresh model with sliding window analysis
experiment_sliding_window_pipeline(
    num_simulations=50000,  # From your parameter sweep
    window_sites=100,       # From your parameter sweep
    num_mice=5,
    num_sessions=10,
    window_size=100,
    stride=25
)

# Option B: Use existing trained model
# 1. Train model using existing script
inference, x_mean, x_std, history, model_dir = experiment_multi_patch_training(num_simulations=50000)

# 2. Simulate data
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_sliding_window import simulate_full_dataset
dataset = simulate_full_dataset(simulator, num_mice=5, num_sessions=10)

# 3. Analyze
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_utils import load_snle_model
_, _, _, _, analysis_dir = load_snle_model(model_dir)
all_results = analyze_dataset(
    simulator, dataset, inference, x_mean, x_std,
    analysis_dir=analysis_dir + '/sliding_window'
)


# ==============================================================================
# TIPS AND BEST PRACTICES
# ==============================================================================

"""
1. ALWAYS use mode='multi' for sliding window analysis
   - Single-patch mode doesn't aggregate properly across windows

2. Match window_size to training data window_sites
   - If trained on 100-site windows, use window_size=100

3. Start with larger stride for quick iteration
   - stride=50 for rapid prototyping
   - stride=25 for standard analysis
   - stride=10 for high temporal resolution

4. Validate on simulated data first
   - Generate data with known parameter evolution
   - Check if inference recovers true parameters
   - Tune hyperparameters before using real data

5. Monitor computation time
   - Each window requires MCMC sampling (~500 samples)
   - 400-site session with stride=25 → ~15 windows
   - Estimate: 30-60 seconds per window

6. Save intermediate results
   - Results contain all posterior samples
   - Can replot without re-running expensive inference
   - torch.save(results, 'results.pt')

7. Check for edge effects
   - First and last windows may have higher uncertainty
   - Fewer complete patches at boundaries
   - Consider trimming or using larger windows at edges

8. Use real data statistics to validate model
   - Compare simulated patches/session to real data
   - Check if inferred parameters produce realistic behavior
   - Refine prior if needed (see README for approaches)
"""