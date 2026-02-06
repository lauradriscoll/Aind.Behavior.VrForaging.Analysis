import os
# Force CPU backend on Apple Silicon to avoid Metal issues
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax.numpy as jnp
from jax import vmap
import jax

# ============================================================================
# Safe utility functions for JAX
# ============================================================================

def _safe_mean(values, mask):
    """Mean of values where mask is True, or 0 if no valid values"""
    masked_values = jnp.where(mask, values, 0.0)
    count = jnp.sum(mask)
    return jnp.where(count > 0, jnp.sum(masked_values) / count, 0.0)

def _safe_std(values, mask):
    """Std of values where mask is True, or 0 if insufficient data"""
    count = jnp.sum(mask)
    mean_val = _safe_mean(values, mask)
    masked_sq_diff = jnp.where(mask, (values - mean_val)**2, 0.0)
    variance = jnp.where(count > 1, jnp.sum(masked_sq_diff) / count, 0.0)
    return jnp.sqrt(variance)

def _safe_corrcoef(x, y, mask):
    """Correlation coefficient, or 0 if insufficient data"""
    count = jnp.sum(mask)
    mean_x = _safe_mean(x, mask)
    mean_y = _safe_mean(y, mask)
    
    cov = _safe_mean((x - mean_x) * (y - mean_y), mask)
    std_x = _safe_std(x, mask)
    std_y = _safe_std(y, mask)
    
    denom = std_x * std_y
    return jnp.where((count > 2) & (denom > 1e-8), cov / denom, 0.0)

def _get_percentile(sorted_arr, p, n_valid):
    """Get percentile from sorted array"""
    idx = jnp.clip(jnp.floor(p * n_valid).astype(jnp.int32), 0, n_valid - 1)
    return jnp.where(n_valid > 0, sorted_arr[idx], 0.0)

# ============================================================================
# Optimized summary statistics (31 features)
# ============================================================================

def compute_summary_stats(window_data):
    """
    Final optimized feature set: 23 features with redundancy removed
    
    Removed 8 redundant features:
    - decision_consistency (r=1.00 with diff_std)
    - reward_effect_magnitude (r=0.999 with mean_time_after_reward)
    - early_exit_rate (r=-0.989 with stop_rate)
    - median (r>0.98 with multiple features)
    - mean_time_after_failure (r>0.96 with many features)
    - prop_patches_with_reward (r>0.95 with time features)
    - within_context_variance (r>0.90 with many features)
    - failure_persistence (r=-0.899 with failure_bump_magnitude)
    """
    
    # Extract columns
    patch_times = window_data[:, 0]
    rewards = window_data[:, 1]
    stops = window_data[:, 2]
    
    # Mask for valid data
    valid_mask = stops > 0
    n_valid = jnp.sum(valid_mask)
    n_sites = len(patch_times)
    
    stats = []
    
    # ================================================================
    # CORE FEATURES (15 features - removed median, mean_time_after_failure, prop_patches_with_reward)
    # ================================================================
    
    # --- Time distribution (2) - REMOVED median, kept IQR ---
    valid_times = jnp.where(valid_mask, patch_times, jnp.inf)
    valid_times_sorted = jnp.sort(valid_times)
    
    p75 = _get_percentile(valid_times_sorted, 0.75, n_valid)
    p25 = _get_percentile(valid_times_sorted, 0.25, n_valid)
    stats.append(p75 - p25)  # iqr (0)
    
    # Early mean (temporal dynamics)
    third = n_sites // 3
    early_mask = valid_mask & (jnp.arange(n_sites) < third)
    early_mean = _safe_mean(patch_times, early_mask)
    stats.append(early_mean)  # early_mean (1)
    
    # --- Reward context (1) - REMOVED mean_time_after_failure, kept mean_time_after_reward ---
    prev_rewards = jnp.roll(rewards, 1).at[0].set(0)
    first_trial_mask = jnp.arange(n_sites) > 0
    after_reward_mask = valid_mask & (prev_rewards > 0) & first_trial_mask
    after_failure_mask = valid_mask & (prev_rewards == 0) & first_trial_mask
    
    mean_after_reward = _safe_mean(patch_times, after_reward_mask)
    mean_after_failure = _safe_mean(patch_times, after_failure_mask)  # Still compute for other features
    
    stats.append(mean_after_reward)  # mean_time_after_reward (2)
    
    # --- Stopping behavior (1) ---
    stats.append(_safe_mean(stops, jnp.ones_like(stops, dtype=bool)))  # stop_rate (3)
    
    # --- Sequential structure (3) ---
    patch_times_t = patch_times[:-1]
    patch_times_t1 = patch_times[1:]
    valid_pairs_mask = valid_mask[:-1] & valid_mask[1:]
    
    stats.append(_safe_corrcoef(patch_times_t, patch_times_t1, valid_pairs_mask))  # autocorr_lag1 (4)
    stats.append(_safe_std(patch_times_t1 - patch_times_t, valid_pairs_mask))  # diff_std (5)
    stats.append(_safe_mean(jnp.abs(patch_times_t1 - patch_times_t), valid_pairs_mask))  # mean_abs_change (6)
    
    # --- Context variability (2) ---
    after_reward_cv = _safe_std(patch_times, after_reward_mask) / jnp.maximum(mean_after_reward, 1e-8)
    after_failure_cv = _safe_std(patch_times, after_failure_mask) / jnp.maximum(mean_after_failure, 1e-8)
    
    stats.append(after_reward_cv)  # after_reward_cv (7)
    stats.append(after_failure_cv)  # after_failure_cv (8)
    
    # --- Signal vs noise (3) ---
    within_reward_var = _safe_std(patch_times, after_reward_mask) ** 2
    within_failure_var = _safe_std(patch_times, after_failure_mask) ** 2
    within_context_variance = (within_reward_var + within_failure_var) / 2  # Compute but don't include
    between_context_variance = (mean_after_failure - mean_after_reward) ** 2
    
    signal_to_noise = between_context_variance / jnp.maximum(within_context_variance, 1e-8)
    stats.append(signal_to_noise)  # signal_to_noise (9)
    
    rewards_t = rewards[:-1]
    reward_predictor = rewards_t.astype(jnp.float32)
    stats.append(_safe_corrcoef(reward_predictor, patch_times_t1, valid_pairs_mask))  # reward_effect_predictability (10)
    
    transition_diffs = jnp.abs(patch_times_t1 - patch_times_t)
    transition_std = _safe_std(transition_diffs, valid_pairs_mask)
    transition_reliability = 1.0 / jnp.maximum(transition_std, 1e-8)
    stats.append(transition_reliability)  # transition_reliability (11)
    
    # --- Temporal dynamics (2) ---
    late_mask = valid_mask & (jnp.arange(n_sites) >= 2*third)
    late_mean = _safe_mean(patch_times, late_mask)
    
    indices = jnp.arange(n_sites, dtype=jnp.float32)
    temporal_trend = _safe_corrcoef(indices, patch_times, valid_mask)
    stats.append(temporal_trend)  # temporal_trend (12)
    stats.append(late_mean - early_mean)  # late_minus_early (13)
    
    # --- Aggregate stats (2) ---
    reward_trial_indices = jnp.arange(n_sites, dtype=jnp.float32)
    mean_reward_trial = _safe_mean(reward_trial_indices, (rewards > 0) & valid_mask)
    stats.append(mean_reward_trial)  # mean_reward_trial (14)
    
    n_patches = jnp.sum(stops == 0) + 1
    mean_sites_per_patch = n_valid / jnp.maximum(n_patches, 1)
    stats.append(mean_sites_per_patch)  # mean_sites_per_patch (15)
    
    # ================================================================
    # DRIFT DISCRIMINATORS (2 features - removed reward_effect_magnitude, early_exit_rate)
    # ================================================================
    
    # 1. First trial time
    first_trial_time = patch_times[0]
    stats.append(first_trial_time)  # first_trial_time (16)
    
    # 2. Reward variability ratio
    std_after_reward = _safe_std(patch_times, after_reward_mask)
    std_overall = _safe_std(patch_times, valid_mask)
    reward_variability_ratio = std_after_reward / jnp.maximum(std_overall, 1e-8)
    stats.append(reward_variability_ratio)  # reward_variability_ratio (17)
    
    # ================================================================
    # NOISE/FAILURE DISCRIMINATORS (6 features - removed within_context_variance, decision_consistency, failure_persistence)
    # ================================================================
    
    baseline_mean = _safe_mean(patch_times, valid_mask)
    std_time = _safe_std(patch_times, valid_mask)
    
    # 1. Local deviation
    times_padded = jnp.pad(patch_times, (1, 1), mode='edge')
    local_means = (times_padded[:-2] + times_padded[2:]) / 2
    local_deviations = jnp.abs(patch_times - local_means)
    local_deviation = _safe_mean(local_deviations, valid_mask)
    stats.append(local_deviation)  # local_deviation (18)
    
    # 2. Coefficient of variation
    cv_overall = std_time / jnp.maximum(baseline_mean, 1e-8)
    stats.append(cv_overall)  # cv_overall (19)
    
    # 3. Autocorrelation at lag 2
    patch_times_t_lag2 = patch_times[:-2]
    patch_times_t2 = patch_times[2:]
    valid_lag2_mask = valid_mask[:-2] & valid_mask[2:]
    autocorr_lag2 = _safe_corrcoef(patch_times_t_lag2, patch_times_t2, valid_lag2_mask)
    stats.append(autocorr_lag2)  # autocorr_lag2 (20)
    
    # 4. Failure bump magnitude
    failure_bump_magnitude = mean_after_reward - mean_after_failure
    stats.append(failure_bump_magnitude)  # failure_bump_magnitude (21)
    
    # 5. Outlier rate
    is_outlier = (jnp.abs(patch_times - baseline_mean) > 2 * std_time) & valid_mask
    outlier_rate = jnp.sum(is_outlier) / jnp.maximum(n_valid, 1.0)
    stats.append(outlier_rate)  # outlier_rate (22)
    
    return jnp.stack(stats)


# Updated feature names (23 features)
FEATURE_NAMES = [
    # Core (15)
    "iqr",                          # 0
    "early_mean",                   # 1
    "mean_time_after_reward",       # 2
    "stop_rate",                    # 3
    "autocorr_lag1",                # 4
    "diff_std",                     # 5
    "mean_abs_change",              # 6
    "after_reward_cv",              # 7
    "after_failure_cv",             # 8
    "signal_to_noise",              # 9
    "reward_effect_predictability", # 10
    "transition_reliability",       # 11
    "temporal_trend",               # 12
    "late_minus_early",             # 13
    "mean_reward_trial",            # 14
    "mean_sites_per_patch",         # 15
    
    # Drift discriminators (2)
    "first_trial_time",             # 16
    "reward_variability_ratio",     # 17
    
    # Noise/Failure discriminators (6)
    "local_deviation",              # 18
    "cv_overall",                   # 19
    "autocorr_lag2",                # 20
    "failure_bump_magnitude",       # 21
    "outlier_rate",                 # 22
]

# ============================================================================
# QUICK TEST
# ============================================================================

def test_stats():
    """Test the final 23-feature optimized statistics function"""
    import numpy as np
    
    # Create test data
    window_data = np.zeros((100, 3))
    window_data[:, 0] = np.random.exponential(2.0, 100)
    window_data[:, 1] = np.random.binomial(1, 0.3, 100)
    window_data[:, 2] = np.random.binomial(1, 0.2, 100)
    
    window_data_jax = jnp.array(window_data)
    stats = compute_summary_stats(window_data_jax)
    
    print(f"✓ Final optimized stats shape: {stats.shape}")
    print(f"✓ Expected shape: (23,)")
    assert stats.shape == (23,), f"Wrong shape: {stats.shape}"
    
    print(f"✓ No NaN values: {not jnp.any(jnp.isnan(stats))}")
    assert not jnp.any(jnp.isnan(stats)), "Found NaN values!"
    
    print(f"✓ No Inf values: {not jnp.any(jnp.isinf(stats))}")
    assert not jnp.any(jnp.isinf(stats)), "Found Inf values!"
    
    print("\nCore features (sample):")
    print(f"  iqr: {stats[0]:.4f}")
    print(f"  early_mean: {stats[1]:.4f}")
    print(f"  mean_time_after_reward: {stats[2]:.4f}")
    print(f"  stop_rate: {stats[3]:.4f}")
    
    print("\nDrift discriminators:")
    print(f"  first_trial_time: {stats[16]:.4f}")
    print(f"  reward_variability_ratio: {stats[17]:.4f}")
    
    print("\nNoise/Failure discriminators:")
    print(f"  local_deviation: {stats[18]:.4f}")
    print(f"  cv_overall: {stats[19]:.4f}")
    print(f"  autocorr_lag2: {stats[20]:.4f}")
    print(f"  failure_bump_magnitude: {stats[21]:.4f}")
    print(f"  outlier_rate: {stats[22]:.4f}")
    
    return True

def analyze_optimized_correlations(n_samples=1000):
    """
    Check correlations in the NEW 23-feature set
    """
    print("="*70)
    print("ANALYZING OPTIMIZED 31-FEATURE CORRELATIONS")
    print("="*70)
    
    from jax import random
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM_JAX, create_prior
    
    rng_key = random.PRNGKey(456)
    simulator = PatchForagingDDM_JAX(max_sites_per_window=100)
    prior_fn = create_prior()
    
    # Generate samples
    print(f"\nGenerating {n_samples} samples from prior...")
    all_stats = []
    for i in range(n_samples):
        if i % 100 == 0:
            print(f"  {i}/{n_samples}...")
        rng_key, subkey1, subkey2 = random.split(rng_key, 3)
        theta = prior_fn().sample(seed=subkey1)['theta']
        window_data, _ = simulator.simulate_one_window(theta, subkey2)
        stats = compute_summary_stats(window_data)
        all_stats.append(stats)
    
    all_stats = jnp.array(all_stats)  # (1000, 31)
    
    # Compute correlation matrix
    corr_matrix = jnp.corrcoef(all_stats.T)
    
    # === PLOT 1: Full correlation heatmap ===
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=90, ha='right', fontsize=7)
    ax.set_yticklabels(FEATURE_NAMES, fontsize=7)
    
    plt.colorbar(im, ax=ax, label='Correlation')
    
    # Add grid lines between groups
    group_boundaries = [0, 15, 17, 23]  # Core (15) | Drift (2) | Noise/Failure (6)
    
    for boundary in group_boundaries:
        ax.axhline(boundary - 0.5, color='black', linewidth=2)
        ax.axvline(boundary - 0.5, color='black', linewidth=2)
    
    plt.title('Optimized 23-Feature Correlation Matrix', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # === IDENTIFY HIGHLY CORRELATED PAIRS ===
    print("\n" + "="*70)
    print("HIGHLY CORRELATED PAIRS (|r| > 0.95)")
    print("="*70)
    
    high_corr_pairs = []
    for i in range(len(FEATURE_NAMES)):
        for j in range(i+1, len(FEATURE_NAMES)):
            if abs(corr_matrix[i, j]) > 0.95:
                high_corr_pairs.append((i, j, corr_matrix[i, j]))
    
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    if len(high_corr_pairs) == 0:
        print("✅ NO highly correlated pairs found! (All |r| < 0.95)")
    else:
        for i, j, r in high_corr_pairs:
            print(f"{FEATURE_NAMES[i]:<22} <-> {FEATURE_NAMES[j]:<22}  r = {r:>6.3f}")
        print(f"\n⚠️  Total highly correlated pairs: {len(high_corr_pairs)}")
    
    # === MODERATE CORRELATIONS (0.85 < |r| < 0.95) ===
    print("\n" + "="*70)
    print("MODERATELY CORRELATED PAIRS (0.85 < |r| < 0.95)")
    print("="*70)
    
    moderate_corr_pairs = []
    for i in range(len(FEATURE_NAMES)):
        for j in range(i+1, len(FEATURE_NAMES)):
            if 0.85 < abs(corr_matrix[i, j]) < 0.95:
                moderate_corr_pairs.append((i, j, corr_matrix[i, j]))
    
    moderate_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    if len(moderate_corr_pairs) == 0:
        print("✅ NO moderately correlated pairs found!")
    else:
        for i, j, r in moderate_corr_pairs[:15]:  # Show top 15
            print(f"{FEATURE_NAMES[i]:<22} <-> {FEATURE_NAMES[j]:<22}  r = {r:>6.3f}")
        if len(moderate_corr_pairs) > 15:
            print(f"... and {len(moderate_corr_pairs) - 15} more")
    
    # === ANALYZE NEW FEATURES SPECIFICALLY ===
    print("\n" + "="*70)
    print("NEW DISCRIMINATOR FEATURES - INTERNAL CORRELATIONS")
    print("="*70)
    
    print("\nDrift vs Reward Discriminators (indices 16-17):")
    drift_reward_indices = [16, 17]
    for i in range(len(drift_reward_indices)):
        for j in range(i+1, len(drift_reward_indices)):
            idx_i = drift_reward_indices[i]
            idx_j = drift_reward_indices[j]
            r = corr_matrix[idx_i, idx_j]
            print(f"  {FEATURE_NAMES[idx_i]:<30} <-> {FEATURE_NAMES[idx_j]:<30}  r = {r:>6.3f}")
    
    print("\nNoise vs Failure Discriminators (indices 18-22):")
    noise_failure_indices = [18, 19, 20, 21, 22]
    
    # Check correlations among noise/failure discriminators (indices 18-22) - UPDATED
    print("\nNoise vs Failure Discriminators (indices 18-22):")
    noise_failure_indices = [18, 19, 20, 21, 22]  # UPDATED for 23 features
    high_internal_corr = []
    for i in range(len(noise_failure_indices)):
        for j in range(i+1, len(noise_failure_indices)):
            idx_i = noise_failure_indices[i]
            idx_j = noise_failure_indices[j]
            r = corr_matrix[idx_i, idx_j]
            if abs(r) > 0.7:  # Flag high internal correlations
                high_internal_corr.append((idx_i, idx_j, r))
                print(f"  ⚠️  {FEATURE_NAMES[idx_i]:<30} <-> {FEATURE_NAMES[idx_j]:<30}  r = {r:>6.3f}")

    if len(high_internal_corr) == 0:
        print("  ✅ No high internal correlations (all |r| < 0.7)")
    
    # === SUMMARY ===
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total features: {len(FEATURE_NAMES)}")
    print(f"Highly correlated pairs (|r| > 0.95): {len(high_corr_pairs)}")
    print(f"Moderately correlated pairs (0.85 < |r| < 0.95): {len(moderate_corr_pairs)}")
    
    if len(high_corr_pairs) == 0 and len(moderate_corr_pairs) < 5:
        print("\n✅ GOOD: Feature set has minimal redundancy!")
    elif len(high_corr_pairs) > 0:
        print(f"\n⚠️  WARNING: {len(high_corr_pairs)} highly redundant pairs - consider removing some")
    else:
        print(f"\n⚠️  MODERATE: {len(moderate_corr_pairs)} moderately correlated pairs - monitor during training")
    
    return all_stats, corr_matrix, high_corr_pairs, moderate_corr_pairs

if __name__ == "__main__":
    print("="*70)
    print("TESTING OPTIMIZED SUMMARY STATISTICS (23 features)")
    print("="*70)
    test_stats()
    
    print("\n" + "="*70)
    print("TESTING PARAMETER SENSITIVITY")
    print("="*70)

    from jax import random
    import jax
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM_JAX, create_prior

    rng_key = random.PRNGKey(123)
    n_test = 1000

    # Setup
    num_window_sites = 100
    simulator = PatchForagingDDM_JAX(max_sites_per_window=num_window_sites)
    prior_fn = create_prior()

    # Collect data
    thetas = []
    stats_list = []

    print(f"Generating {n_test} samples from prior...")
    for i in range(n_test):
        if i % 100 == 0:
            print(f"  {i}/{n_test}...")
        rng_key, subkey1, subkey2 = random.split(rng_key, 3)
        theta = prior_fn().sample(seed=subkey1)['theta']
        window_data, _ = simulator.simulate_one_window(theta, subkey2)
        stats = compute_summary_stats(window_data)
        
        thetas.append(theta)
        stats_list.append(stats)

    thetas = jnp.array(thetas)
    stats_list = jnp.array(stats_list)

    # Check correlations
    param_names = ["drift_rate", "reward_bump", "failure_bump", "noise_std"]
    
    print("\n" + "="*70)
    print("PARAMETER-STATISTIC CORRELATIONS")
    print("="*70)
    
    print("\nDrift vs Reward Discriminators:")
    drift_reward_indices = [16, 17]  # UPDATED: first_trial_time, reward_variability_ratio
    drift_reward_names = ["first_trial_time", "reward_var_ratio"]

    print(f"{'Param':<15}", end='')
    for name in drift_reward_names:
        print(f"{name[:18]:>20}", end='')
    print()
    print("-" * 55)

    for i, pname in enumerate(param_names):
        print(f"{pname:<15}", end='')
        for idx in drift_reward_indices:
            corr = jnp.corrcoef(thetas[:, i], stats_list[:, idx])[0, 1]
            print(f"{corr:>20.3f}", end='')
        print()

    print("\nNoise vs Failure Discriminators:")
    noise_failure_indices = [18, 19, 20, 21, 22]  # UPDATED: local_dev, cv_overall, autocorr2, fail_mag, outlier_rate
    noise_failure_names = ["local_dev", "cv_overall", "autocorr2", "fail_mag", "outlier_rate"]

    print(f"{'Param':<15}", end='')
    for name in noise_failure_names:
        print(f"{name[:12]:>14}", end='')
    print()
    print("-" * 85)

    for i, pname in enumerate(param_names):
        print(f"{pname:<15}", end='')
        for idx in noise_failure_indices:
            corr = jnp.corrcoef(thetas[:, i], stats_list[:, idx])[0, 1]
            print(f"{corr:>14.3f}", end='')
        print()

    print("\n" + "="*70)
    print("KEY CORRELATIONS:")
    print("="*70)
    print(f"drift_rate vs first_trial_time:         r={jnp.corrcoef(thetas[:, 0], stats_list[:, 16])[0, 1]:.3f}")
    print(f"reward_bump vs reward_var_ratio:         r={jnp.corrcoef(thetas[:, 1], stats_list[:, 17])[0, 1]:.3f}")
    print(f"failure_bump vs failure_bump_mag:        r={jnp.corrcoef(thetas[:, 2], stats_list[:, 21])[0, 1]:.3f}")
    print(f"noise_std vs local_deviation:            r={jnp.corrcoef(thetas[:, 3], stats_list[:, 18])[0, 1]:.3f}")
    print(f"noise_std vs cv_overall:                 r={jnp.corrcoef(thetas[:, 3], stats_list[:, 19])[0, 1]:.3f}")
    print(f"noise_std vs outlier_rate:               r={jnp.corrcoef(thetas[:, 3], stats_list[:, 22])[0, 1]:.3f}")

    # Also update core features
    print("\nCore features correlations:")
    print(f"drift_rate vs stop_rate:                 r={jnp.corrcoef(thetas[:, 0], stats_list[:, 3])[0, 1]:.3f}")
    print(f"drift_rate vs diff_std:                  r={jnp.corrcoef(thetas[:, 0], stats_list[:, 5])[0, 1]:.3f}")
    print(f"reward_bump vs mean_time_after_reward:   r={jnp.corrcoef(thetas[:, 1], stats_list[:, 2])[0, 1]:.3f}")
    print(f"reward_bump vs autocorr_lag1:            r={jnp.corrcoef(thetas[:, 1], stats_list[:, 4])[0, 1]:.3f}")
        
    all_stats_opt, corr_matrix_opt, high_corr, moderate_corr = analyze_optimized_correlations(n_samples=1000)
    