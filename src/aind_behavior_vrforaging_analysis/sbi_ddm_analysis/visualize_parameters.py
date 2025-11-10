"""
Visualization tool for understanding parameter effects on behavior.

Creates grid plots showing how different parameter combinations
affect the simulated behavioral trajectories.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM


def plot_single_window(ax, window, theta, title=None):
    """
    Plot a single window showing behavioral trajectory.
    
    Args:
        ax: matplotlib axis
        window: (100, 3) tensor [time, reward, stopped]
        theta: (3,) tensor [drift_rate, reward_bump, failure_bump]
        title: optional title
    """
    times = window[:, 0].numpy()
    rewards = window[:, 1].numpy()
    stopped = window[:, 2].numpy()
    
    # Find patch boundaries (where stopped=0)
    leave_indices = np.where(stopped == 0)[0]
    
    # Color code by outcome
    colors = []
    for i in range(len(window)):
        if stopped[i] == 0:
            colors.append('red')  # Left patch
        elif rewards[i] == 1:
            colors.append('green')  # Got reward
        else:
            colors.append('orange')  # No reward
    
    # Plot trajectory
    ax.scatter(range(len(window)), times, c=colors, s=20, alpha=0.7)
    
    # Add vertical lines at patch boundaries
    for leave_idx in leave_indices:
        ax.axvline(leave_idx, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    # Labels and formatting
    ax.set_xlabel('Site Index', fontsize=8)
    ax.set_ylabel('Time in Patch', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    
    if title:
        ax.set_title(title, fontsize=9)
    
    # Add parameter info
    param_text = f'drift={theta[0]:.2f}\nreward={theta[1]:.2f}\nfailure={theta[2]:.2f}'
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
            fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add legend (small)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Reward'),
        Patch(facecolor='orange', alpha=0.7, label='No reward'),
        Patch(facecolor='red', alpha=0.7, label='Left patch')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=6)


def plot_parameter_grid_2d(
    param1_name: str,
    param2_name: str,
    param1_range: tuple,
    param2_range: tuple,
    fixed_param_value: float,
    grid_size: int = 5,
    save_path: str = 'parameter_grid.png'
):
    """
    Create grid of behavioral trajectories across 2D parameter space.
    
    Args:
        param1_name: 'drift_rate', 'reward_bump', or 'failure_bump'
        param2_name: 'drift_rate', 'reward_bump', or 'failure_bump'
        param1_range: (min, max) for param1
        param2_range: (min, max) for param2
        fixed_param_value: Value for the third parameter
        grid_size: Number of values per dimension (e.g., 5 = 5x5 grid)
        save_path: Where to save the plot
    """
    param_names = ['drift_rate', 'reward_bump', 'failure_bump']
    param_indices = {name: i for i, name in enumerate(param_names)}
    
    # Get indices
    idx1 = param_indices[param1_name]
    idx2 = param_indices[param2_name]
    fixed_idx = [i for i in range(3) if i not in [idx1, idx2]][0]
    fixed_name = param_names[fixed_idx]
    
    # Create parameter grids
    param1_values = np.linspace(param1_range[0], param1_range[1], grid_size)
    param2_values = np.linspace(param2_range[0], param2_range[1], grid_size)
    
    # Initialize simulator
    simulator = PatchForagingDDM()

    # Create figure
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(grid_size, grid_size, figure=fig, hspace=0.3, wspace=0.3)
    
    print(f"\nGenerating {grid_size}x{grid_size} parameter grid...")
    print(f"  {param1_name}: {param1_range}")
    print(f"  {param2_name}: {param2_range}")
    print(f"  {fixed_name}: {fixed_param_value} (fixed)")
    
    # Generate trajectories for each parameter combination
    for i, p1_val in enumerate(param1_values):
        for j, p2_val in enumerate(param2_values):
            # Construct theta
            theta = torch.zeros(3)
            theta[idx1] = p1_val
            theta[idx2] = p2_val
            theta[fixed_idx] = fixed_param_value
            
            # Simulate window
            window, _ = simulator.simulate_with_walk(
                theta_mean=theta,
                window_sites=100,
                sigma=0.01  # Small random walk for clearer visualization
            )
            
            # Plot
            ax = fig.add_subplot(gs[grid_size-1-j, i])  # Flip j for standard orientation
            
            # Add title only on edges
            title = None
            if j == grid_size - 1:  # Top row
                title = f'{param1_name}={p1_val:.2f}'
            
            plot_single_window(ax, window, theta, title=title)
            
            # Add y-axis label on left column
            if i == 0:
                ax.set_ylabel(f'{param2_name}={p2_val:.2f}\nTime in Patch', fontsize=8)
    
    # Overall title
    fig.suptitle(
        f'Behavioral Trajectories Across Parameter Space\n'
        f'{param1_name} vs {param2_name} (fixed {fixed_name}={fixed_param_value:.2f})',
        fontsize=14, fontweight='bold'
    )
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nGrid plot saved to {save_path}")
    plt.close()


def plot_all_parameter_combinations(
    grid_size: int = 5,
    save_dir: str = 'analysis_results/visualize_parameters/w_noise'
):
    """
    Create all three 2D parameter grids.

    Generates:
    1. drift_rate vs reward_bump (fixed failure_bump)
    2. drift_rate vs failure_bump (fixed reward_bump)
    3. reward_bump vs failure_bump (fixed drift_rate)
    """

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    print("="*60)
    print("Generating Parameter Space Visualizations")
    print("="*60)

    # Grid 1: drift_rate vs reward_bump
    print("\n1. drift_rate vs reward_bump")
    plot_parameter_grid_2d(
        param1_name='drift_rate',
        param2_name='reward_bump',
        param1_range=(0.1, 1.5),
        param2_range=(0.1, 1.5),
        fixed_param_value=0.3,  # failure_bump
        grid_size=grid_size,
        save_path=f'{save_dir}/grid_drift_vs_reward.png'
    )
    
    # Grid 2: drift_rate vs failure_bump
    print("\n2. drift_rate vs failure_bump")
    plot_parameter_grid_2d(
        param1_name='drift_rate',
        param2_name='failure_bump',
        param1_range=(0.1, 1.5),
        param2_range=(0.1, 1.5),
        fixed_param_value=0.6,  # reward_bump
        grid_size=grid_size,
        save_path=f'{save_dir}/grid_drift_vs_failure.png'
    )
    
    # Grid 3: reward_bump vs failure_bump
    print("\n3. reward_bump vs failure_bump")
    plot_parameter_grid_2d(
        param1_name='reward_bump',
        param2_name='failure_bump',
        param1_range=(0.1, 1.5),
        param2_range=(0.1, 1.5),
        fixed_param_value=0.5,  # drift_rate
        grid_size=grid_size,
        save_path=f'{save_dir}/grid_reward_vs_failure.png'
    )
    
    print("\n" + "="*60)
    print("All grids generated!")
    print("="*60)


def plot_parameter_effect_summary(save_dir: str = 'analysis_results/visualize_parameters/w_noise'):
    """
    Create summary figure showing effect of each parameter individually.
    """

    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/parameter_effects.png'

    simulator = PatchForagingDDM()
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    param_names = ['drift_rate', 'reward_bump', 'failure_bump']
    param_ranges = [(0.2, 1.5), (0.2, 1.5), (0.0, 1.5)]
    base_theta = torch.tensor([0.5, 0.6, 0.3])
    
    print("\nGenerating parameter effect summary...")
    
    for param_idx, (param_name, param_range) in enumerate(zip(param_names, param_ranges)):
        param_values = np.linspace(param_range[0], param_range[1], 3)
        
        for i, param_val in enumerate(param_values):
            # Create theta with one parameter varied
            theta = base_theta.clone()
            theta[param_idx] = param_val
            
            # Simulate
            window, _, _ = simulator.simulate_window_with_random_walk(
                theta_mean=theta,
                window_sites=100,
                random_walk_sigma=0.01
            )
            
            # Plot
            ax = axes[param_idx, i]
            plot_single_window(ax, window, theta, 
                             title=f'{param_name}={param_val:.2f}')
    
    fig.suptitle('Effect of Each Parameter on Behavior', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Summary plot saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        # Quick summary figure
        plot_parameter_effect_summary()

        print("\nVisualization complete!")
        print(f"Generated:")
        print(f"  - parameter_effects.png")
    else:
        # Full grid analysis
        grid_size = 5
        if len(sys.argv) > 1:
            grid_size = int(sys.argv[1])
        
        plot_all_parameter_combinations(grid_size=grid_size)
        
        print("\nVisualization complete!")
        print(f"Generated:")
        print(f"  - grid_drift_vs_reward.png")
        print(f"  - grid_drift_vs_failure.png")
        print(f"  - grid_reward_vs_failure.png")
        print(f"  - parameter_effects.png")