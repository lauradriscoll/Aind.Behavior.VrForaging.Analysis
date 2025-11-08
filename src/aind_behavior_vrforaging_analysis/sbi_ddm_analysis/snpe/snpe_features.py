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
    Output: (604,) feature vector
    
    Features:
    - Original data flattened (300)
    - Inter-site intervals (100)
    - Cumulative rewards per patch (100)
    - Failure run lengths (100)
    - Summary statistics (4 features)
    """
    times = window[:, 0]
    rewards = window[:, 1]
    stopped = window[:, 2]
    
    features = []
    
    # === Original data (300 features) ===
    features.append(window.flatten())
    
    # === Inter-site intervals (100 features) ===
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
    cumulative_rewards = torch.zeros_like(rewards)
    reward_count = 0
    for i in range(len(rewards)):
        if patch_boundaries[i] > 0:
            reward_count = 0
        cumulative_rewards[i] = reward_count
        if stopped[i] > 0 and rewards[i] > 0:
            reward_count += 1
    features.append(cumulative_rewards)
    
    # === Cumulative failures (100 features) ==
    cumulative_failures = torch.zeros_like(rewards)
    failure_count = 0
    for i in range(len(rewards)):
        if patch_boundaries[i] > 0:
            failure_count = 0
        cumulative_failures[i] = failure_count
        if stopped[i] > 0 and rewards[i] == 0:
            failure_count += 1
    features.append(cumulative_failures)

    # === Summary statistics (4 features) ===
    summary = torch.tensor([
        times.max()/100,
        rewards.mean(),
        stopped.mean(),
        (cumulative_failures > 0).float().mean(),
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