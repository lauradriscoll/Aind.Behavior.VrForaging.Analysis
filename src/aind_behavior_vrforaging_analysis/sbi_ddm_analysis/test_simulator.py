import torch
import numpy as np
import matplotlib.pyplot as plt
from simulator import PatchForagingDDM, reward_probability

def test_basic_simulation():
    """Test basic simulator functionality"""
    print("="*60)
    print("TEST 1: Basic Simulation")
    print("="*60)
    
    simulator = PatchForagingDDM()
    
    # Test parameters: [drift_rate, reward_bump, failure_bump]
    theta = torch.tensor([0.3, 0.4, 0.15])
    
    # Simulate trial
    data = simulator(theta, max_sites=50)
    
    print(f"Data shape: {data.shape}")  # Should be (50, 3)
    print(f"\nFirst 20 sites:")
    print("   time | reward | stopped")
    for i in range(20):
        print(f"{i+1:3d}: {data[i, 0]:6.2f} | {int(data[i, 1]):6d} | {int(data[i, 2]):7d}")
    
    # Find patch boundaries
    left_indices = (data[:, 2] == 0).nonzero(as_tuple=True)[0]
    print(f"\nPatch boundaries (left at sites): {left_indices.tolist()}")
    print(f"Number of patches visited: {len(left_indices)}")
    
    # Check that time resets after leaving
    if len(left_indices) > 0:
        first_leave = left_indices[0].item()
        if first_leave < len(data) - 1:
            print(f"\nTime at leave (site {first_leave+1}): {data[first_leave, 0]:.2f}")
            print(f"Time at next site (site {first_leave+2}): {data[first_leave+1, 0]:.2f}")
            print("✓ Time should reset to small value after leaving")
    
    return data


def test_parameter_effects():
    """Test how different parameters affect behavior"""
    print("\n" + "="*60)
    print("TEST 2: Parameter Effects")
    print("="*60)
    
    simulator = PatchForagingDDM()
    max_sites = 100
    
    # Test different parameter combinations
    test_cases = [
        ("Low drift, high reward bump", torch.tensor([0.1, 0.6, 0.1])),
        ("High drift, low reward bump", torch.tensor([0.6, 0.2, 0.1])),
        ("High failure bump", torch.tensor([0.3, 0.4, 0.5])),
        ("Low failure bump", torch.tensor([0.3, 0.4, 0.05])),
    ]
    
    for name, theta in test_cases:
        data = simulator(theta, max_sites=max_sites)
        
        # Calculate statistics
        n_patches = (data[:, 2] == 0).sum().item()
        total_rewards = data[:, 1].sum().item()
        avg_patch_length = max_sites / n_patches if n_patches > 0 else max_sites
        
        print(f"\n{name}")
        print(f"  Parameters: drift={theta[0]:.2f}, rew_bump={theta[1]:.2f}, fail_bump={theta[2]:.2f}")
        print(f"  Patches visited: {n_patches}")
        print(f"  Avg sites per patch: {avg_patch_length:.1f}")
        print(f"  Total rewards: {total_rewards}")


def test_reward_probability_decay():
    """Test that reward probability decays as expected"""
    print("\n" + "="*60)
    print("TEST 3: Reward Probability Decay")
    print("="*60)
    
    initial_prob = 0.8
    decay_rate = -0.1
    
    print(f"Initial prob: {initial_prob}, Decay rate: {decay_rate}")
    print("\nReward # | Probability")
    print("-" * 25)
    for n in range(10):
        prob = reward_probability(n, initial_prob, decay_rate)
        print(f"{n:8d} | {prob:.4f}")


def visualize_single_trial(theta=None):
    """Visualize a single trial"""
    print("\n" + "="*60)
    print("TEST 4: Visualization")
    print("="*60)
    
    if theta is None:
        theta = torch.tensor([0.3, 0.4, 0.15])
    
    simulator = PatchForagingDDM()
    data = simulator(theta, max_sites=100)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Panel 1: Time course with rewards and leaves
    ax = axes[0]
    times = data[:, 0].numpy()
    rewards = data[:, 1].numpy()
    stopped = data[:, 2].numpy()
    
    # Plot timeline
    site_indices = np.arange(len(data))
    ax.scatter(site_indices[rewards == 1], times[rewards == 1], 
              c='green', s=100, marker='o', label='Reward', zorder=3)
    ax.scatter(site_indices[rewards == 0], times[rewards == 0], 
              c='gray', s=50, marker='x', label='No reward', alpha=0.5, zorder=2)
    ax.scatter(site_indices[stopped == 0], times[stopped == 0], 
              c='red', s=150, marker='s', label='Left patch', zorder=4)
    
    ax.set_xlabel('Site number', fontsize=12)
    ax.set_ylabel('Time since patch start (s)', fontsize=12)
    ax.set_title('Trial Timeline', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Patch durations
    ax = axes[1]
    leave_indices = np.where(stopped == 0)[0]
    patch_durations = times[leave_indices]
    
    if len(patch_durations) > 0:
        ax.bar(range(len(patch_durations)), patch_durations, color='steelblue', alpha=0.7)
        ax.set_xlabel('Patch number', fontsize=12)
        ax.set_ylabel('Patch duration (s)', fontsize=12)
        ax.set_title(f'Patch Durations (mean={patch_durations.mean():.2f}s)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Rewards per patch
    ax = axes[2]
    if len(leave_indices) > 0:
        patch_starts = [0] + (leave_indices + 1).tolist()
        patch_ends = leave_indices.tolist() + [len(data)]
        
        rewards_per_patch = []
        for start, end in zip(patch_starts, patch_ends):
            if start < len(data):
                rewards_per_patch.append(rewards[start:end].sum())
        
        ax.bar(range(len(rewards_per_patch)), rewards_per_patch, 
              color='green', alpha=0.7)
        ax.set_xlabel('Patch number', fontsize=12)
        ax.set_ylabel('Number of rewards', fontsize=12)
        ax.set_title(f'Rewards per Patch (mean={np.mean(rewards_per_patch):.2f})', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('simulator_test.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'simulator_test.png'")
    plt.show()


def test_batch_simulation():
    """Test batch simulation"""
    print("\n" + "="*60)
    print("TEST 5: Batch Simulation")
    print("="*60)
    
    simulator = PatchForagingDDM()
    
    # Batch of parameters
    theta_batch = torch.tensor([
        [0.2, 0.3, 0.1],
        [0.4, 0.5, 0.2],
        [0.3, 0.4, 0.15],
    ])
    
    data_batch = simulator(theta_batch, max_sites=50)
    
    print(f"Batch shape: {data_batch.shape}")  # Should be (3, 50, 3)
    print(f"✓ Batch simulation works!")


if __name__ == "__main__":
    # Run all tests
    test_basic_simulation()
    test_parameter_effects()
    test_reward_probability_decay()
    test_batch_simulation()
    visualize_single_trial()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)