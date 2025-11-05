"""
Feature engineering for patch foraging windows.

Single purpose: Transform (100, 3) raw behavioral data into feature vector.
"""

import torch
import numpy as np


def extract_features(window: torch.Tensor) -> torch.Tensor:
    """
    Extract features from behavioral window.
    
    Input: (100, 3) window [time_since_patch_start, reward, stopped]
    Output: (611,) feature vector
    
    Features:
    - Original data flattened (300)
    - Inter-site intervals (100)
    - Cumulative rewards per patch (100)
    - Failure run lengths (100)
    - Summary statistics (11)
    """
    times = window[:, 0]
    rewards = window[:, 1]
    stopped = window[:, 2]
    
    features = []
    
    # === Original data (300 features) ===
    features.append(window.flatten())
    
    # === Inter-site intervals (100 features) ===
    # Key for identifying drift_rate!
    intervals = torch.cat([times[0:1], times[1:] - times[:-1]])
    
    # Detect patch boundaries: where animal left (stopped=0)
    patch_boundaries = torch.cat([
        torch.zeros(1),
        (stopped[:-1] == 0).float()
    ])
    
    # Zero out intervals at patch boundaries
    intervals_masked = intervals.clone()
    intervals_masked[patch_boundaries.bool()] = 0
    features.append(intervals_masked)
    
    # === Cumulative rewards (100 features) ===
    # Rewards collected before each site (affects reward probability)
    cumulative_rewards = torch.zeros_like(rewards)
    reward_count = 0
    for i in range(len(rewards)):
        if patch_boundaries[i] > 0:
            reward_count = 0
        cumulative_rewards[i] = reward_count
        if stopped[i] > 0 and rewards[i] > 0:
            reward_count += 1
    features.append(cumulative_rewards)
    
    # === Failure runs (100 features) ===
    # Consecutive failures - isolates drift_rate (no reward_bump acting)
    failure_run = torch.zeros_like(rewards)
    run_length = 0
    for i in range(len(rewards)):
        if patch_boundaries[i] > 0:
            run_length = 0
        if stopped[i] > 0:
            if rewards[i] == 0:
                run_length += 1
            else:
                run_length = 0
        else:
            run_length = 0
        failure_run[i] = run_length
    features.append(failure_run)
    
    # === Summary statistics (11 features) ===
    # Robust aggregate features
    valid_intervals = intervals_masked[intervals_masked > 0]
    mean_interval = valid_intervals.mean() if len(valid_intervals) > 0 else torch.tensor(0.0)
    std_interval = valid_intervals.std() if len(valid_intervals) > 1 else torch.tensor(0.0)
    
    summary = torch.tensor([
        mean_interval,
        std_interval,
        times.max(),
        rewards.sum(),
        rewards.mean(),
        cumulative_rewards.max(),
        stopped.sum(),
        stopped.mean(),
        (stopped == 0).sum().float(),
        failure_run.max(),
        (failure_run > 0).float().mean(),
    ])
    summary = torch.nan_to_num(summary, nan=0.0, posinf=0.0, neginf=0.0)
    features.append(summary)
    
    # Concatenate all features
    return torch.cat([f.flatten() for f in features])


# Test
if __name__ == "__main__":
    print("Testing feature extraction...")
    
    # Create dummy window
    window = torch.randn(100, 3)
    window[:, 1] = torch.bernoulli(torch.ones(100) * 0.3)  # rewards
    window[:, 2] = torch.bernoulli(torch.ones(100) * 0.8)  # stopped
    window[:, 0] = window[:, 0].abs()  # times
    
    features = extract_features(window)
    
    print(f"Input shape: {window.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected: (611,)")
    print(f"✓ Test passed!" if features.shape[0] == 611 else "✗ Test failed!")