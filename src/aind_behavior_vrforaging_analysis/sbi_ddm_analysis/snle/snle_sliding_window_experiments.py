"""
Experiment: Full sliding window analysis pipeline.

Purpose: Complete workflow from SNLE training to multi-mouse session analysis.
"""

###################################################################################
noise_std = 0.0 #true for sweep_20251108_202608
###################################################################################

import torch
import os
from datetime import datetime


def experiment_sliding_window_pipeline(
    num_simulations: int = 2000000,
    window_sites: int = 100,
    num_mice: int = 5,
    num_sessions: int = 10,
    sites_per_session: tuple = (350, 400),
    window_size: int = 100,
    stride: int = 25,
    drift_increase: float = 0.3,
    analysis_dir: str = None,
    model_dir: str = '/Users/laura.driscoll/Documents/code/Aind.Behavior.VrForaging.Analysis/src/aind_behavior_vrforaging_analysis/sbi_ddm_analysis/snle/snle_models/multi_patch_20251111_224417/'
):
    """
    Complete pipeline: Train SNLE → Simulate data → Analyze all sessions.
    
    Args:
        num_simulations: Training samples for SNLE
        window_sites: Sites per training simulation
        num_mice: Number of mice to simulate
        num_sessions: Sessions per mouse
        sites_per_session: (min, max) sites in each session
        window_size: Sites per inference window
        stride: Sites between consecutive windows
        drift_increase: How much drift rate increases over session
        analysis_dir: Directory for outputs (auto-generated if None)
        model_dir: Location of trained SNLE model (if exists, will load instead of training)
    
    Returns:
        dataset: Simulated data
        all_results: Inference results for all mice/sessions
        model_dir: Location of trained SNLE model
    """
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM, create_prior
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_inference import train_snle
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_utils import save_snle_model, load_snle_model
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_sliding_window import (
        simulate_full_dataset, analyze_dataset
    )
    
    print("="*80)
    print("SLIDING WINDOW ANALYSIS PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Training: {num_simulations} simulations, {window_sites} sites/sim")
    print(f"  Dataset: {num_mice} mice × {num_sessions} sessions")
    print(f"  Sessions: {sites_per_session[0]}-{sites_per_session[1]} sites each")
    print(f"  Analysis: {window_size}-site windows, stride={stride}")
    print(f"  Drift increase: +{drift_increase} over session")
    print("="*80)

    # Load or train SNLE model
    simulator = PatchForagingDDM(noise_std=noise_std)
    prior = create_prior()

    if model_dir is not None and os.path.exists(model_dir):
        # Model exists → load it
        print("\nLoading existing SNLE model...")
        model_path = os.path.join(model_dir)
        inference, x_mean, x_std, mode, analysis_dir = load_snle_model(model_path)
    else:
        # No model → train a new one
        print("\nNo existing model found. Training new SNLE model...")

        # Create analysis directory if not provided
        if analysis_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_dir = f"sliding_window_analysis_{timestamp}"
        os.makedirs(analysis_dir, exist_ok=True)
        print(f"\nAnalysis directory: {analysis_dir}")

        # Step 1: Train SNLE model
        print("\n" + "="*80)
        print("STEP 1: Training SNLE Model (multi-patch mode)")
        print("="*80)
        
        likelihood_estimator, inference, x_mean, x_std, history = train_snle(
            simulator, prior,
            num_simulations=num_simulations,
            window_sites=window_sites,
            mode='multi',
            max_num_epochs=50,
            batch_size=50
        )
        
        # Save model
        model_dir = save_snle_model(inference, x_mean, x_std, mode='multi')
        print(f"\nModel saved to: {model_dir}")

        
        # Step 2: Simulate multi-mouse dataset
        print("\n" + "="*80)
        print("STEP 2: Simulating Multi-Mouse Dataset")
        print("="*80)
    
    dataset = simulate_full_dataset(
        simulator,
        num_mice=num_mice,
        num_sessions=num_sessions,
        sites_per_session=sites_per_session,
        base_params=torch.tensor([0.2, 0.5, 0.2]),
        param_std=0.05,
        drift_increase=drift_increase
    )
    
    print("\nDataset statistics:")
    for mouse_id in dataset['mice']:
        evolution = dataset['evolution_types'][mouse_id]
        num_sites_per_session = [
            len(dataset['data'][mouse_id][sid][0]) 
            for sid in range(num_sessions)
        ]
        avg_sites = sum(num_sites_per_session) / len(num_sites_per_session)
        print(f"  Mouse {mouse_id} ({evolution:8s}): avg {avg_sites:.0f} sites/session")
    
    # Step 3: Analyze all sessions
    print("\n" + "="*80)
    print("STEP 3: Running Sliding Window Analysis")
    print("="*80)
    
    all_results = analyze_dataset(
        simulator, dataset,
        inference, x_mean, x_std,
        window_size=window_size,
        stride=stride,
        num_samples=500,
        analysis_dir=analysis_dir
    )
    
    # Step 4: Summary statistics
    print("\n" + "="*80)
    print("STEP 4: Computing Summary Statistics")
    print("="*80)
    
    compute_summary_statistics(dataset, all_results, analysis_dir)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {analysis_dir}")
    print(f"Model saved to: {model_dir}")
    
    return dataset, all_results, model_dir


def compute_summary_statistics(dataset, all_results, analysis_dir):
    """
    Compute and save summary statistics across all mice/sessions.
    
    Args:
        dataset: Simulated dataset
        all_results: Inference results
        analysis_dir: Directory to save summary
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("\nComputing drift rate evolution statistics...")
    
    # Collect drift rate changes across all sessions
    drift_changes_gradual = []
    drift_changes_stepwise = []
    
    for mouse_id in dataset['mice']:
        evolution_type = dataset['evolution_types'][mouse_id]
        
        for session_id in all_results[mouse_id].keys():
            results = all_results[mouse_id][session_id]
            drift_means = results['posterior_means'][:, 0].numpy()
            
            if len(drift_means) > 0:
                drift_change = drift_means[-1] - drift_means[0]
                
                if evolution_type == 'gradual':
                    drift_changes_gradual.append(drift_change)
                else:
                    drift_changes_stepwise.append(drift_change)
    
    # Print statistics
    print(f"\nDrift rate change over sessions:")
    if drift_changes_gradual:
        print(f"  Gradual mice:  mean={np.mean(drift_changes_gradual):.3f} ± {np.std(drift_changes_gradual):.3f}")
    if drift_changes_stepwise:
        print(f"  Stepwise mice: mean={np.mean(drift_changes_stepwise):.3f} ± {np.std(drift_changes_stepwise):.3f}")
    
    # Plot distribution of drift changes
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    if drift_changes_gradual:
        ax.hist(drift_changes_gradual, bins=15, alpha=0.6, label='Gradual', color='blue')
    if drift_changes_stepwise:
        ax.hist(drift_changes_stepwise, bins=15, alpha=0.6, label='Stepwise', color='red')
    
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Change in Drift Rate (first → last window)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Drift Rate Changes Across Sessions', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_path = os.path.join(analysis_dir, 'drift_change_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved distribution plot to: {save_path}")
    
    # Save summary to text file
    summary_path = os.path.join(analysis_dir, 'summary_statistics.txt')
    with open(summary_path, 'w') as f:
        f.write("SLIDING WINDOW ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total mice: {len(dataset['mice'])}\n")
        f.write(f"Sessions per mouse: {len(dataset['data'][0])}\n\n")
        
        f.write("Evolution types:\n")
        for mouse_id in dataset['mice']:
            f.write(f"  Mouse {mouse_id}: {dataset['evolution_types'][mouse_id]}\n")
        
        f.write("\nDrift rate change statistics:\n")
        if drift_changes_gradual:
            f.write(f"  Gradual mice:  {np.mean(drift_changes_gradual):.3f} ± {np.std(drift_changes_gradual):.3f}\n")
        if drift_changes_stepwise:
            f.write(f"  Stepwise mice: {np.mean(drift_changes_stepwise):.3f} ± {np.std(drift_changes_stepwise):.3f}\n")
    
    print(f"  Saved summary to: {summary_path}")


def experiment_quick_test():
    """
    Quick test with small dataset for debugging.
    """
    print("Running quick test with small dataset...")
    
    return experiment_sliding_window_pipeline(
        num_simulations=2000,
        window_sites=100,
        num_mice=2,
        num_sessions=2,
        sites_per_session=(150, 200),
        window_size=100,
        stride=25,
        drift_increase=0.2,
        analysis_dir='test_sliding_window'
    )


def experiment_single_session_demo():
    """
    Demonstrate sliding window on a single session with detailed plots.
    """
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM, create_prior
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_inference import train_snle
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_sliding_window import (
        simulate_session_with_evolution, sliding_window_inference, plot_parameter_evolution
    )
    
    print("="*80)
    print("SINGLE SESSION DEMONSTRATION")
    print("="*80)
    
    # Setup
    simulator = PatchForagingDDM()
    prior = create_prior()
    
    # Train model
    print("\n1. Training SNLE...")
    _, inference, x_mean, x_std, _ = train_snle(
        simulator, prior,
        num_simulations=10000,
        window_sites=100,
        mode='multi',
        max_num_epochs=20
    )
    
    # Simulate session with gradual drift increase
    print("\n2. Simulating session with gradual drift increase...")
    base_params = torch.tensor([0.2, 0.5, 0.2])
    session_data, true_params = simulate_session_with_evolution(
        simulator,
        base_params=base_params,
        num_sites=400,
        evolution_type='gradual',
        drift_increase=0.3
    )
    print(f"   Generated {len(session_data)} sites")
    
    # Run sliding window analysis
    print("\n3. Running sliding window inference...")
    results = sliding_window_inference(
        simulator, session_data,
        inference, x_mean, x_std,
        window_size=100,
        stride=10,
        num_samples=500,
        warmup_steps=100
    )
    
    # Plot
    print("\n4. Generating plot...")
    plot_parameter_evolution(results, true_params, 
                           save_path='single_session_demo.png')
    
    print("\n" + "="*80)
    print("Demo complete! Check 'single_session_demo.png'")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            # Quick test
            experiment_quick_test()
        
        elif sys.argv[1] == 'demo':
            # Single session demo
            experiment_single_session_demo()
        
        elif sys.argv[1] == 'full':
            # Full pipeline with custom parameters
            num_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 2000000
            experiment_sliding_window_pipeline(num_simulations=num_sims)
        
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python snle_sliding_window_experiments.py [test|demo|full] [num_simulations]")
    
    else:
        # Default: run quick test
        print("No arguments provided. Running quick test...")
        print("Use 'test', 'demo', or 'full' for different experiments.\n")
        experiment_quick_test()