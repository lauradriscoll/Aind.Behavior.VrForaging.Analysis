"""
Test correlation of enhanced 35-feature set with parameters.
Focus on validating mechanistic features for failure_bump and noise_std.
"""
import os
# Force CPU backend on Apple Silicon to avoid Metal issues
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax.numpy as jnp
from jax import random
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# Use enhanced stats
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.enhanced_stats_35 import FEATURE_NAMES, compute_summary_stats


def validate_enhanced_features(simulator, prior_fn, n_samples=2000):
    """
    Test if new features improve correlation with failure_bump and noise_std
    """
    print("="*70)
    print("VALIDATING ENHANCED FEATURES")
    print("="*70)
    print(f"\nGenerating {n_samples} samples...")
    
    rng_key = random.PRNGKey(42)
    thetas = []
    stats = []
    
    for i in range(n_samples):
        if i % 200 == 0:
            print(f"  {i}/{n_samples}...")
        
        rng_key, subkey1, subkey2 = random.split(rng_key, 3)
        theta = prior_fn().sample(seed=subkey1)['theta']
        window_data, _ = simulator.simulate_one_window(theta, subkey2)
        
        summary_stats = compute_summary_stats(window_data)
        
        thetas.append(theta)
        stats.append(summary_stats)
    
    thetas = jnp.array(thetas)
    stats = jnp.array(stats)
    
    param_names = ['drift_rate', 'reward_bump', 'failure_bump', 'noise_std']
    
    # =========================================================================
    # ANALYZE EACH PARAMETER
    # =========================================================================
    
    results = {}
    
    for i, param in enumerate(param_names):
        print(f"\n{'='*70}")
        print(f"{param.upper()}")
        print('='*70)
        
        # Compute correlations for all features
        correlations = []
        for j in range(35):
            r, p = spearmanr(np.array(thetas[:, i]), np.array(stats[:, j]))
            correlations.append({
                'feature_idx': j,
                'feature_name': FEATURE_NAMES[j],
                'correlation': r,
                'abs_correlation': abs(r),
                'p_value': p
            })
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        results[param] = correlations
        
        # Print top 10
        print(f"\nTop 10 features for {param}:")
        print(f"  {'Rank':<6} {'Feature':<35} {'Correlation':>12} {'p-value':>10}")
        print("  " + "-"*65)
        for rank, c in enumerate(correlations[:10], 1):
            print(f"  {rank:<6} {c['feature_name']:<35} {c['correlation']:>12.3f} {c['p_value']:>10.3e}")
    
    # =========================================================================
    # FOCUSED ANALYSIS: NEW FEATURES
    # =========================================================================
    
    print("\n" + "="*70)
    print("IMPACT OF NEW FEATURES")
    print("="*70)
    
    # For failure_bump
    print("\n--- FAILURE_BUMP ---")
    print("\nBaseline (23-feat best):")
    baseline_failure = next(c for c in results['failure_bump'] if c['feature_name'] == 'failure_bump_magnitude')
    print(f"  failure_bump_magnitude: ρ = {baseline_failure['correlation']:.3f}")
    
    print("\nNew mechanistic features:")
    mechanistic_failure_names = [
        'immediate_failure_effect',
        'failure_decay',
        'failure_sensitivity',
        'failure_streak_effect',
        'failure_reward_cv_asymmetry'
    ]
    
    for fname in mechanistic_failure_names:
        feat = next(c for c in results['failure_bump'] if c['feature_name'] == fname)
        improvement = abs(feat['correlation']) - abs(baseline_failure['correlation'])
        symbol = "✅" if improvement > 0 else "❌"
        print(f"  {symbol} {fname:<35} ρ = {feat['correlation']:>6.3f}  "
              f"(Δ = {improvement:>+6.3f})")
    
    # For noise_std
    print("\n--- NOISE_STD ---")
    print("\nBaseline (23-feat best):")
    baseline_noise = next(c for c in results['noise_std'] if c['feature_name'] == 'outlier_rate')
    print(f"  outlier_rate: ρ = {baseline_noise['correlation']:.3f}")
    
    print("\nNew mechanistic features:")
    mechanistic_noise_names = [
        'detrended_variance',
        'neighbor_inconsistency',
        'high_freq_power',
        'range_iqr_ratio'
    ]
    
    for fname in mechanistic_noise_names:
        feat = next(c for c in results['noise_std'] if c['feature_name'] == fname)
        improvement = abs(feat['correlation']) - abs(baseline_noise['correlation'])
        symbol = "✅" if improvement > 0 else "❌"
        print(f"  {symbol} {fname:<35} ρ = {feat['correlation']:>6.3f}  "
              f"(Δ = {improvement:>+6.3f})")
    
    # For reward_bump (strategic additions)
    print("\n--- REWARD_BUMP ---")
    print("\nBaseline (23-feat best):")
    baseline_reward = results['reward_bump'][0]  # Already sorted
    print(f"  {baseline_reward['feature_name']}: ρ = {baseline_reward['correlation']:.3f}")
    
    print("\nStrategic additions:")
    strategic_names = ['median', 'std_time', 'max_time']
    
    for fname in strategic_names:
        feat = next(c for c in results['reward_bump'] if c['feature_name'] == fname)
        improvement = abs(feat['correlation']) - abs(baseline_reward['correlation'])
        symbol = "✅" if improvement > 0 else "❌"
        print(f"  {symbol} {fname:<35} ρ = {feat['correlation']:>6.3f}  "
              f"(Δ = {improvement:>+6.3f})")
    
    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    
    print("\n" + "="*70)
    print("SUMMARY: BEST SINGLE FEATURE FOR EACH PARAMETER")
    print("="*70)
    
    for param in param_names:
        best = results[param][0]
        print(f"\n{param}:")
        print(f"  Best feature: {best['feature_name']}")
        print(f"  Correlation:  ρ = {best['correlation']:.3f}")
        
        # Check if it's a new feature
        is_new = best['feature_idx'] >= 23
        if is_new:
            print(f"  ✨ NEW FEATURE! (index {best['feature_idx']})")
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for idx, (ax, param) in enumerate(zip(axes.flat, param_names)):
        # Get top 3 features
        top_3 = results[param][:3]
        
        # Plot scatter for best feature
        best_feat_idx = top_3[0]['feature_idx']
        best_feat_name = top_3[0]['feature_name']
        
        ax.scatter(stats[:, best_feat_idx], thetas[:, idx], 
                  alpha=0.3, s=10, c='blue')
        ax.set_xlabel(best_feat_name, fontsize=12)
        ax.set_ylabel(param, fontsize=12)
        ax.set_title(f'{param}\nBest: {best_feat_name} (ρ={top_3[0]["correlation"]:.3f})',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add text with top 3
        text = "Top 3 features:\n"
        for i, feat in enumerate(top_3, 1):
            text += f"{i}. {feat['feature_name'][:20]}... (ρ={feat['correlation']:.3f})\n"
        
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('enhanced_features_validation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved: enhanced_features_validation.png")
    
    return results


# =============================================================================
# RUN VALIDATION
# =============================================================================

if __name__ == "__main__":
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import (
        PatchForagingDDM_JAX, create_prior
    )
    
    # Setup
    simulator = PatchForagingDDM_JAX(max_sites_per_window=100)
    prior_fn = create_prior()
    
    # Make sure simulator uses enhanced stats
    print("NOTE: Update your simulator to use compute_enhanced_summary_stats!")
    print("      Check that simulator._compute_summary_stats calls the enhanced version.\n")
    
    # Run validation
    results = validate_enhanced_features(simulator, prior_fn, n_samples=2000)
    
    print("\n" + "="*70)
    print("✓ VALIDATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. If new features show improvement → Retrain SNLE with 35 features")
    print("2. If improvement is marginal → Consider using just strategic additions (26 features)")
    print("3. Re-run SBC after retraining to check calibration")