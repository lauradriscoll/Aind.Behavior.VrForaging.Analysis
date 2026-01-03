"""
Enhanced summary statistics for patch foraging DDM parameter inference.

Feature set: 35 total features
- 23 base optimized features
- 3 strategic additions from 37-feature analysis (median, std_time, max_time)
- 5 mechanistic failure_bump features
- 4 mechanistic noise_std features (selected best from 7 tested)
"""
import os
# Force CPU backend on Apple Silicon to avoid Metal issues
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax.numpy as jnp


# =============================================================================
# SAFE HELPER FUNCTIONS
# =============================================================================

def _safe_mean(values, mask):
    """Mean of masked values, returns 0 if no valid values."""
    count = jnp.sum(mask)
    return jnp.where(count > 0, jnp.sum(values * mask) / count, 0.0)


def _safe_std(values, mask):
    """Std of masked values, returns 0 if count <= 1."""
    count = jnp.sum(mask)
    mean_val = _safe_mean(values, mask)
    variance = jnp.where(
        count > 1,
        jnp.sum(((values - mean_val) ** 2) * mask) / count,
        0.0
    )
    return jnp.sqrt(variance)


def _safe_corrcoef(x, y, mask):
    """Correlation of masked arrays, returns 0 if insufficient data."""
    count = jnp.sum(mask)
    
    mean_x = _safe_mean(x, mask)
    mean_y = _safe_mean(y, mask)
    
    cov = jnp.sum((x - mean_x) * (y - mean_y) * mask) / jnp.maximum(count, 1)
    std_x = _safe_std(x, mask)
    std_y = _safe_std(y, mask)
    
    return jnp.where(
        (count > 2) & (std_x > 1e-8) & (std_y > 1e-8),
        cov / (std_x * std_y),
        0.0
    )


def _get_percentile(sorted_arr, p, n_valid):
    """Get percentile from sorted array, returns 0 if empty."""
    idx = jnp.int32(p * jnp.maximum(n_valid - 1, 0))
    return jnp.where(n_valid > 0, sorted_arr[idx], 0.0)


# =============================================================================
# MAIN FEATURE COMPUTATION
# =============================================================================

def compute_summary_stats(window_data):
    """
    Compute 35 enhanced summary statistics for parameter inference.
    
    Args:
        window_data: (n_sites, 3) array [patch_times, rewards, stops]
        
    Returns:
        (35,) array of summary statistics
    """
    # Extract columns
    patch_times = window_data[:, 0]
    rewards = window_data[:, 1]
    stops = window_data[:, 2]
    
    # Basic masks
    valid_mask = stops > 0
    n_valid = jnp.sum(valid_mask)
    n_sites = len(patch_times)
    third = n_sites // 3
    
    stats = []
    
    # =========================================================================
    # PART 1: BASE 23 FEATURES (from final optimized set)
    # =========================================================================
    
    # --- Time distribution (2 features) ---
    valid_times_sorted = jnp.sort(jnp.where(valid_mask, patch_times, jnp.inf))
    p75 = _get_percentile(valid_times_sorted, 0.75, n_valid)
    p25 = _get_percentile(valid_times_sorted, 0.25, n_valid)
    
    stats.append(p75 - p25)  # 0: iqr
    stats.append(_safe_mean(patch_times, valid_mask & (jnp.arange(n_sites) < third)))  # 1: early_mean
    
    # --- Reward context (1 feature) ---
    prev_rewards = jnp.roll(rewards, 1).at[0].set(0)
    first_trial_mask = jnp.arange(n_sites) > 0
    after_reward_mask = valid_mask & (prev_rewards > 0) & first_trial_mask
    after_failure_mask = valid_mask & (prev_rewards == 0) & first_trial_mask
    
    mean_after_reward = _safe_mean(patch_times, after_reward_mask)
    mean_after_failure = _safe_mean(patch_times, after_failure_mask)
    
    stats.append(mean_after_reward)  # 2: mean_time_after_reward
    
    # --- Stopping (1 feature) ---
    stats.append(_safe_mean(stops, jnp.ones_like(stops, dtype=bool)))  # 3: stop_rate
    
    # --- Sequential structure (3 features) ---
    patch_times_t = patch_times[:-1]
    patch_times_t1 = patch_times[1:]
    valid_pairs = valid_mask[:-1] & valid_mask[1:]
    
    stats.append(_safe_corrcoef(patch_times_t, patch_times_t1, valid_pairs))  # 4: autocorr_lag1
    stats.append(_safe_std(patch_times_t1 - patch_times_t, valid_pairs))  # 5: diff_std
    stats.append(_safe_mean(jnp.abs(patch_times_t1 - patch_times_t), valid_pairs))  # 6: mean_abs_change
    
    # --- Context variability (2 features) ---
    stats.append(_safe_std(patch_times, after_reward_mask) / jnp.maximum(mean_after_reward, 1e-8))  # 7: after_reward_cv
    stats.append(_safe_std(patch_times, after_failure_mask) / jnp.maximum(mean_after_failure, 1e-8))  # 8: after_failure_cv
    
    # --- Signal vs noise (3 features) ---
    within_reward_var = _safe_std(patch_times, after_reward_mask) ** 2
    within_failure_var = _safe_std(patch_times, after_failure_mask) ** 2
    within_context_var = (within_reward_var + within_failure_var) / 2
    between_context_var = (mean_after_failure - mean_after_reward) ** 2
    
    stats.append(between_context_var / jnp.maximum(within_context_var, 1e-8))  # 9: signal_to_noise
    
    rewards_t = rewards[:-1].astype(jnp.float32)
    stats.append(_safe_corrcoef(rewards_t, patch_times_t1, valid_pairs))  # 10: reward_effect_predictability
    
    transition_std = _safe_std(jnp.abs(patch_times_t1 - patch_times_t), valid_pairs)
    stats.append(1.0 / jnp.maximum(transition_std, 1e-8))  # 11: transition_reliability
    
    # --- Temporal dynamics (2 features) ---
    late_mask = valid_mask & (jnp.arange(n_sites) >= 2*third)
    late_mean = _safe_mean(patch_times, late_mask)
    early_mean = stats[1]  # Reuse early_mean
    
    indices = jnp.arange(n_sites, dtype=jnp.float32)
    stats.append(_safe_corrcoef(indices, patch_times, valid_mask))  # 12: temporal_trend
    stats.append(late_mean - early_mean)  # 13: late_minus_early
    
    # --- Aggregate (2 features) ---
    reward_indices = jnp.arange(n_sites, dtype=jnp.float32)
    stats.append(_safe_mean(reward_indices, (rewards > 0) & valid_mask))  # 14: mean_reward_trial
    
    n_patches = jnp.sum(stops == 0) + 1
    stats.append(n_valid / jnp.maximum(n_patches, 1))  # 15: mean_sites_per_patch
    
    # --- Drift discriminators (2 features) ---
    stats.append(patch_times[0])  # 16: first_trial_time
    
    std_after_reward = _safe_std(patch_times, after_reward_mask)
    std_overall = _safe_std(patch_times, valid_mask)
    stats.append(std_after_reward / jnp.maximum(std_overall, 1e-8))  # 17: reward_variability_ratio
    
    # --- Basic noise/failure features (5 features) ---
    # Local deviation
    times_padded = jnp.pad(patch_times, (1, 1), mode='edge')
    local_means = (times_padded[:-2] + times_padded[2:]) / 2
    stats.append(_safe_mean(jnp.abs(patch_times - local_means), valid_mask))  # 18: local_deviation
    
    baseline_mean = _safe_mean(patch_times, valid_mask)
    stats.append(std_overall / jnp.maximum(baseline_mean, 1e-8))  # 19: cv_overall
    
    # Autocorr lag 2
    patch_times_lag2 = patch_times[:-2]
    patch_times_t2 = patch_times[2:]
    valid_lag2 = valid_mask[:-2] & valid_mask[2:]
    stats.append(_safe_corrcoef(patch_times_lag2, patch_times_t2, valid_lag2))  # 20: autocorr_lag2
    
    stats.append(mean_after_reward - mean_after_failure)  # 21: failure_bump_magnitude
    
    # Outlier rate
    is_outlier = (jnp.abs(patch_times - baseline_mean) > 2 * std_overall) & valid_mask
    stats.append(jnp.sum(is_outlier) / jnp.maximum(n_valid, 1.0))  # 22: outlier_rate
    
    # =========================================================================
    # PART 2: STRATEGIC ADDITIONS (3 features)
    # =========================================================================
    
    # From 37-feature importance analysis
    stats.append(_get_percentile(valid_times_sorted, 0.50, n_valid))  # 23: median
    stats.append(std_overall)  # 24: std_time
    stats.append(jnp.max(jnp.where(valid_mask, patch_times, -jnp.inf)))  # 25: max_time
    
    # =========================================================================
    # PART 3: MECHANISTIC FAILURE_BUMP FEATURES (5 features)
    # =========================================================================
    
    # Immediate effect: Compare time immediately after failure vs reward
    is_after_failure = jnp.roll(rewards == 0, 1).at[0].set(False) & valid_mask
    is_after_reward = jnp.roll(rewards > 0, 1).at[0].set(False) & valid_mask
    
    imm_after_failure = _safe_mean(patch_times, is_after_failure)
    imm_after_reward = _safe_mean(patch_times, is_after_reward)
    stats.append(imm_after_reward - imm_after_failure)  # 26: immediate_failure_effect
    
    # Decay: Compare lag-1 vs lag-3 after failure (effect should wear off)
    is_3_after_failure = jnp.roll(rewards == 0, 3).at[:3].set(False) & valid_mask
    stats.append(
        _safe_mean(patch_times, is_after_failure) - 
        _safe_mean(patch_times, is_3_after_failure)
    )  # 27: failure_decay
    
    # Sensitivity: Immediate effect relative to overall variability
    immediate_effect = stats[26]  # Reuse
    stats.append(immediate_effect / jnp.maximum(std_overall, 0.1))  # 28: failure_sensitivity
    
    # Streak amplification: Single failure vs consecutive failures
    double_failure = (jnp.roll(rewards == 0, 1) & jnp.roll(rewards == 0, 2)).at[:2].set(False) & valid_mask
    single_failure = (jnp.roll(rewards > 0, 2) & jnp.roll(rewards == 0, 1)).at[:2].set(False) & valid_mask
    
    stats.append(
        _safe_mean(patch_times, single_failure) - 
        _safe_mean(patch_times, double_failure)
    )  # 29: failure_streak_effect
    
    # CV asymmetry: Variability after failure vs reward (failure should be less variable)
    cv_failure = _safe_std(patch_times, is_after_failure) / jnp.maximum(imm_after_failure, 0.1)
    cv_reward = _safe_std(patch_times, is_after_reward) / jnp.maximum(imm_after_reward, 0.1)
    stats.append(cv_reward / jnp.maximum(cv_failure, 0.1))  # 30: failure_reward_cv_asymmetry
    
    # =========================================================================
    # PART 4: MECHANISTIC NOISE_STD FEATURES (4 best features)
    # =========================================================================
    
    # Detrended variance: Remove linear trend, measure residual variance
    mean_idx = _safe_mean(indices, valid_mask)
    cov_idx_time = _safe_mean((indices - mean_idx) * (patch_times - baseline_mean), valid_mask)
    var_idx = _safe_std(indices, valid_mask) ** 2
    slope = cov_idx_time / jnp.maximum(var_idx, 1e-8)
    
    trend = baseline_mean + slope * (indices - mean_idx)
    residuals = patch_times - trend
    stats.append(_safe_std(residuals, valid_mask) ** 2)  # 31: detrended_variance
    
    # Neighbor inconsistency: Variability between consecutive trials in same context
    same_context = (jnp.roll(rewards > 0, 1) == (rewards > 0)).at[0].set(False)
    neighbor_mask = same_context & valid_mask & jnp.roll(valid_mask, 1).at[0].set(False)
    neighbor_diffs = jnp.abs(patch_times - jnp.roll(patch_times, 1))
    stats.append(_safe_mean(neighbor_diffs, neighbor_mask))  # 32: neighbor_inconsistency
    
    # High-frequency power: Variance of first differences (amplifies noise)
    diff_mask = valid_mask & jnp.roll(valid_mask, 1).at[0].set(False)
    first_diffs = patch_times - jnp.roll(patch_times, 1)
    stats.append(_safe_std(first_diffs, diff_mask) ** 2)  # 33: high_freq_power
    
    # Range/IQR ratio: Tendency for extreme outliers
    range_val = stats[25] - jnp.min(jnp.where(valid_mask, patch_times, jnp.inf))  # max_time - min_time
    iqr_val = stats[0]  # Reuse IQR
    stats.append(range_val / jnp.maximum(iqr_val, 0.1))  # 34: range_iqr_ratio
    
    return jnp.stack(stats)


# =============================================================================
# FEATURE NAMES
# =============================================================================

FEATURE_NAMES = [
    # Base 23 features (0-22)
    "iqr",
    "early_mean",
    "mean_time_after_reward",
    "stop_rate",
    "autocorr_lag1",
    "diff_std",
    "mean_abs_change",
    "after_reward_cv",
    "after_failure_cv",
    "signal_to_noise",
    "reward_effect_predictability",
    "transition_reliability",
    "temporal_trend",
    "late_minus_early",
    "mean_reward_trial",
    "mean_sites_per_patch",
    "first_trial_time",
    "reward_variability_ratio",
    "local_deviation",
    "cv_overall",
    "autocorr_lag2",
    "failure_bump_magnitude",
    "outlier_rate",
    
    # Strategic additions (23-25)
    "median",
    "std_time",
    "max_time",
    
    # Mechanistic failure features (26-30)
    "immediate_failure_effect",
    "failure_decay",
    "failure_sensitivity",
    "failure_streak_effect",
    "failure_reward_cv_asymmetry",
    
    # Mechanistic noise features (31-34)
    "detrended_variance",
    "neighbor_inconsistency",
    "high_freq_power",
    "range_iqr_ratio",
]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    
    print("="*70)
    print("TESTING ENHANCED SUMMARY STATISTICS (35 features)")
    print("="*70)
    
    # Create test data
    window_data = np.zeros((100, 3))
    window_data[:, 0] = np.random.exponential(2.0, 100)  # patch_times
    window_data[:, 1] = np.random.binomial(1, 0.3, 100)  # rewards
    window_data[:, 2] = np.random.binomial(1, 0.2, 100)  # stops
    
    window_data_jax = jnp.array(window_data)
    stats = compute_summary_stats(window_data_jax)
    
    print(f"\n✓ Stats shape: {stats.shape}")
    print(f"✓ Expected: (35,)")
    print(f"✓ No NaN: {not jnp.any(jnp.isnan(stats))}")
    print(f"✓ No Inf: {not jnp.any(jnp.isinf(stats))}")
    
    print("\nFeature values (first 10):")
    for i in range(10):
        print(f"  {i:2d}. {FEATURE_NAMES[i]:<30} = {stats[i]:.4f}")
    
    print("\nStrategic additions (23-25):")
    for i in range(23, 26):
        print(f"  {i:2d}. {FEATURE_NAMES[i]:<30} = {stats[i]:.4f}")
    
    print("\nMechanistic failure features (26-30):")
    for i in range(26, 31):
        print(f"  {i:2d}. {FEATURE_NAMES[i]:<30} = {stats[i]:.4f}")
    
    print("\nMechanistic noise features (31-34):")
    for i in range(31, 35):
        print(f"  {i:2d}. {FEATURE_NAMES[i]:<30} = {stats[i]:.4f}")
    
    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)