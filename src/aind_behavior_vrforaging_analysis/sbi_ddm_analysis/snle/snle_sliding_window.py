"""
Sliding window SNLE inference for session-level parameter evolution.

Purpose: Analyze how DDM parameters change over time within behavioral sessions.
Supports realistic multi-mouse simulation with individual differences.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional


def simulate_mouse_parameters(mouse_id: int, num_sessions: int = 10, 
                              base_params: Optional[torch.Tensor] = None,
                              theta_ranges: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                              param_std: float = 0.05) -> torch.Tensor:
    """
    Generate mouse-specific base parameters for all sessions.
    
    Args:
        mouse_id: Mouse identifier (0-4)
        num_sessions: Number of sessions per mouse
        base_params: Base [drift_rate, reward_bump, failure_bump] or None for default
        theta_ranges: (low, high) parameter bounds for each of the three parameters
        param_std: Between-session variability
    
    Returns:
        params: (num_sessions, 3) tensor of session-level base parameters
    """

    if theta_ranges is None:
        low = torch.tensor([0.4, 0.05, 0.05])
        high = torch.tensor([.6, .9, .9])
        theta_ranges = (low, high)
    
    if base_params is None:
        base_params = torch.tensor([0.2, 0.5, 0.2])
    
    # Each mouse has slightly different baseline
    mouse_offset = torch.randn(3) * 0.1
    mouse_base = base_params + mouse_offset
    
    # Add session-to-session variability
    session_noise = torch.randn(num_sessions, 3) * param_std
    session_params = mouse_base.unsqueeze(0) + session_noise
    
    # Ensure positive parameters
    session_params = torch.clamp(session_params, min=0.01)
    
    return session_params


def simulate_session_with_evolution(simulator, base_params: torch.Tensor,
                                   num_sites: int = 400,
                                   evolution_type: str = 'gradual',
                                   drift_increase: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate a single session where parameters evolve over time.
    
    Args:
        simulator: PatchForagingDDM instance
        base_params: (3,) Starting [drift_rate, reward_bump, failure_bump]
        num_sites: Total sites in session (350-400)
        evolution_type: 'constant', 'gradual', or 'stepwise'
        drift_increase: How much drift_rate increases by end of session
    
    Returns:
        data: (num_sites, 3) behavioral data [time, num_stops, num_rewards]
        true_params: (num_sites, 3) true parameters at each site
    """
    drift_rate, reward_bump, failure_bump = base_params
    
    
    if evolution_type == 'constant':
        param_gen = simulator.walk_params(base_params, sigma=0, shift=0)
    
    elif evolution_type == 'gradual':
        param_gen = simulator.walk_params(base_params, sigma=0, shift=drift_increase)

    elif evolution_type == 'stepwise':
        param_gen = simulator.step_change_params(2, theta_ranges = [base_params, base_params+torch.tensor([drift_increase, 0.0, 0.0])])
    else:
        raise ValueError(f"Unknown evolution_type: {evolution_type}")
    
    # Simulate session data
    session_data, _, true_params = simulator.simulate_trial(param_gen, window_sites=num_sites, return_aggregate=True)
    
    return session_data, np.array(true_params)


def simulate_full_dataset(simulator, num_mice: int = 5, num_sessions: int = 10,
                         sites_per_session: Tuple[int, int] = (350, 400),
                         base_params: Optional[torch.Tensor] = None,
                         param_std: float = 0.05,
                         drift_increase: float = 0.3) -> Dict:
    """
    Simulate complete multi-mouse, multi-session dataset.
    
    Args:
        simulator: PatchForagingDDM instance
        num_mice: Number of mice
        num_sessions: Sessions per mouse
        sites_per_session: (min, max) sites per session
        base_params: Population mean parameters
        param_std: Between-session std
        drift_increase: Drift rate increase over session
    
    Returns:
        dataset: Dict with structure:
            'mice': List of mouse_ids
            'data': Dict[mouse_id][session_id] = (data, true_params)
            'evolution_types': Dict[mouse_id] = 'gradual' or 'stepwise'
            'base_params': Dict[mouse_id][session_id] = base parameters
    """
    print(f"Simulating dataset: {num_mice} mice × {num_sessions} sessions")
    
    if base_params is None:
        base_params = torch.tensor([0.2, 0.5, 0.2])
    
    # Assign evolution types to mice (some gradual, some stepwise)
    evolution_types = {}
    for mouse_id in range(num_mice):
        evolution_types[mouse_id] = 'gradual' if mouse_id % 2 == 0 else 'stepwise'
    
    dataset = {
        'mice': list(range(num_mice)),
        'data': {},
        'evolution_types': evolution_types,
        'base_params': {},
    }
    
    for mouse_id in tqdm(range(num_mice), desc="Simulating mice"):
        # Generate mouse-specific session parameters
        session_base_params = simulate_mouse_parameters(
            mouse_id, num_sessions, base_params, param_std
        )
        
        dataset['data'][mouse_id] = {}
        dataset['base_params'][mouse_id] = {}
        
        evolution_type = evolution_types[mouse_id]
        
        for session_id in range(num_sessions):
            # Random session length
            num_sites = np.random.randint(*sites_per_session)
            
            # Simulate session
            data, true_params = simulate_session_with_evolution(
                simulator,
                base_params=session_base_params[session_id],
                num_sites=num_sites,
                evolution_type=evolution_type,
                drift_increase=drift_increase
            )
            
            dataset['data'][mouse_id][session_id] = (data, true_params)
            dataset['base_params'][mouse_id][session_id] = session_base_params[session_id]
    
    print(f"\nDataset simulation complete!")
    print(f"  Evolution types: {evolution_types}")
    print(f"  Sites per session: {sites_per_session[0]}-{sites_per_session[1]}")
    
    return dataset


def sliding_window_inference(simulator, session_data: torch.Tensor,
                            inference, x_mean: torch.Tensor, x_std: torch.Tensor,
                            window_size: int = 100, stride: int = 25,
                            num_samples: int = 500, warmup_steps: int = 100) -> Dict:
    """
    Apply SNLE inference over sliding windows in a session.
    
    Args:
        simulator: PatchForagingDDM instance
        session_data: (num_sites, 3) behavioral data
        inference: Trained SNLE model
        x_mean, x_std: Normalization parameters
        window_size: Sites per window
        stride: Step size between windows
        num_samples: Posterior samples per window
        warmup_steps: MCMC warmup
    
    Returns:
        results: Dict with:
            'window_centers': Window center positions
            'posterior_samples': List of (num_samples, 3) tensors
            'posterior_means': (num_windows, 3) means
            'posterior_stds': (num_windows, 3) stds
    """
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_inference import infer_parameters_snle
    
    num_sites = len(session_data)
    num_windows = (num_sites - window_size) // stride + 1
    
    print(f"Running sliding window inference:")
    print(f"  Session sites: {num_sites}")
    print(f"  Window size: {window_size}")
    print(f"  Stride: {stride}")
    print(f"  Number of windows: {num_windows}")
    
    window_centers = []
    posterior_samples_list = []
    posterior_means = []
    posterior_stds = []
    
    for window_idx in tqdm(range(num_windows), desc="Sliding windows"):
        start_idx = window_idx * stride
        end_idx = start_idx + window_size
        
        # Extract window
        window_data = session_data[start_idx:end_idx]
        
        # Compute summary statistics for this window (multi-patch mode)
        # Need to aggregate across patches in window
        # For multi-patch mode, we need mean/std of per-patch stats
        
        # First, identify patches in this window
        patch_times = []
        patch_stops = []
        patch_rewards = []
        
        current_patch_time = 0
        current_patch_stops = 0
        current_patch_rewards = 0
        
        for site_data in window_data:
            time, stops, rewards = site_data
            current_patch_time += time
            current_patch_stops += stops
            current_patch_rewards += rewards
            
            # Check if left patch (reward = 0 indicates leaving)
            if rewards == 0 and current_patch_time > 0:
                patch_times.append(current_patch_time)
                patch_stops.append(current_patch_stops)
                patch_rewards.append(current_patch_rewards)
                
                # Reset for next patch
                current_patch_time = 0
                current_patch_stops = 0
                current_patch_rewards = 0
        
        # Add final patch if still in one
        if current_patch_time > 0:
            patch_times.append(current_patch_time)
            patch_stops.append(current_patch_stops)
            patch_rewards.append(current_patch_rewards)
        
        # Compute multi-patch summary stats
        if len(patch_times) == 0:
            # Edge case: no complete patches in window
            # Use dummy stats (will likely have high uncertainty)
            observed_stats = torch.zeros(8)
        else:
            patch_times = torch.tensor(patch_times, dtype=torch.float32)
            patch_stops = torch.tensor(patch_stops, dtype=torch.float32)
            patch_rewards = torch.tensor(patch_rewards, dtype=torch.float32)
            
            observed_stats = torch.tensor([
                patch_times.mean().item(),
                patch_times.std().item() if len(patch_times) > 1 else 0.0,
                patch_stops.mean().item(),
                patch_stops.std().item() if len(patch_stops) > 1 else 0.0,
                patch_rewards.mean().item(),
                patch_rewards.std().item() if len(patch_rewards) > 1 else 0.0,
                patch_rewards.sum().item(),
                float(len(patch_times))
            ])
        
        # Run inference
        posterior_samples = infer_parameters_snle(
            inference, observed_stats,
            x_mean, x_std,
            num_samples=num_samples,
            warmup_steps=warmup_steps
        )
        
        window_center = start_idx + window_size // 2
        window_centers.append(window_center)
        posterior_samples_list.append(posterior_samples)
        posterior_means.append(posterior_samples.mean(dim=0))
        posterior_stds.append(posterior_samples.std(dim=0))
    
    results = {
        'window_centers': torch.tensor(window_centers),
        'posterior_samples': posterior_samples_list,
        'posterior_means': torch.stack(posterior_means),
        'posterior_stds': torch.stack(posterior_stds),
        'num_windows': num_windows,
    }
    
    return results


def plot_parameter_evolution(results: Dict, true_params: Optional[torch.Tensor] = None,
                            session_id: Optional[int] = None,
                            mouse_id: Optional[int] = None,
                            save_path: Optional[str] = None):
    """
    Plot parameter evolution over session with uncertainty bands.
    
    Args:
        results: Output from sliding_window_inference
        true_params: (num_sites, 3) true parameters if available
        session_id: Session identifier for title
        mouse_id: Mouse identifier for title
        save_path: Path to save figure
    """
    param_names = ['Drift Rate', 'Reward Bump', 'Failure Bump']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    window_centers = results['window_centers'].numpy()
    means = results['posterior_means'].numpy()
    stds = results['posterior_stds'].numpy()
    
    for param_idx, (ax, param_name) in enumerate(zip(axes, param_names)):
        # Plot posterior mean with uncertainty
        ax.plot(window_centers, means[:, param_idx], 
               'o-', linewidth=2, markersize=4, label='Posterior Mean')
        ax.fill_between(window_centers,
                        means[:, param_idx] - stds[:, param_idx],
                        means[:, param_idx] + stds[:, param_idx],
                        alpha=0.3, label='±1 SD')
        
        # Plot true parameters if available
        print(true_params)
        if true_params is not None:
            # Subsample true params to match window centers for visibility
            ax.plot(range(len(true_params)), true_params[:, param_idx],
                   'k--', alpha=0.5, linewidth=1.5, label='True Parameter')
        
        ax.set_ylabel(param_name, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        if param_idx == 0:
            title = f"Parameter Evolution Over Session"
            if mouse_id is not None:
                title += f" (Mouse {mouse_id}"
                if session_id is not None:
                    title += f", Session {session_id}"
                title += ")"
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        if param_idx == 2:
            ax.set_xlabel('Site in Session', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_multi_session_comparison(all_results: List[Dict], 
                                  all_true_params: List[torch.Tensor],
                                  mouse_id: int,
                                  save_path: Optional[str] = None):
    """
    Plot parameter evolution across multiple sessions for one mouse.
    
    Args:
        all_results: List of results dicts from sliding_window_inference
        all_true_params: List of true parameter tensors
        mouse_id: Mouse identifier
        save_path: Path to save figure
    """
    param_names = ['Drift Rate', 'Reward Bump', 'Failure Bump']
    num_sessions = len(all_results)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, num_sessions))
    
    for param_idx, (ax, param_name) in enumerate(zip(axes, param_names)):
        for session_idx, (results, true_params) in enumerate(zip(all_results, all_true_params)):
            window_centers = results['window_centers'].numpy()
            means = results['posterior_means'].numpy()
            
            # Offset x-axis by session
            x_offset = session_idx * 500  # Space sessions apart
            
            ax.plot(window_centers + x_offset, means[:, param_idx],
                   'o-', color=colors[session_idx], alpha=0.7,
                   linewidth=1.5, markersize=3, label=f'Session {session_idx}')
        
        ax.set_ylabel(param_name, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if param_idx == 0:
            ax.set_title(f"Mouse {mouse_id}: Parameter Evolution Across Sessions",
                        fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
        
        if param_idx == 2:
            ax.set_xlabel('Site (offset by session)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_dataset(simulator, dataset: Dict, inference, x_mean: torch.Tensor, x_std: torch.Tensor,
                   window_size: int = 100, stride: int = 25,
                   num_samples: int = 500, analysis_dir: Optional[str] = None):
    """
    Run sliding window analysis on entire simulated dataset.
    
    Args:
        simulator: PatchForagingDDM instance
        dataset: Output from simulate_full_dataset
        inference: Trained SNLE model
        x_mean, x_std: Normalization parameters
        window_size: Sites per window
        stride: Step size between windows
        num_samples: Posterior samples per window
        analysis_dir: Directory to save plots
    
    Returns:
        all_results: Dict[mouse_id][session_id] = inference results
    """
    import os
    
    if analysis_dir:
        os.makedirs(analysis_dir, exist_ok=True)
    
    all_results = {}
    
    for mouse_id in dataset['mice']:
        print(f"\n{'='*60}")
        print(f"Analyzing Mouse {mouse_id} ({dataset['evolution_types'][mouse_id]} evolution)")
        print(f"{'='*60}")
        
        all_results[mouse_id] = {}
        session_results = []
        session_true_params = []
        
        for session_id in range(len(dataset['data'][mouse_id])):
            print(f"\nSession {session_id}:")
            
            session_data, true_params = dataset['data'][mouse_id][session_id]
            
            # Run sliding window inference
            results = sliding_window_inference(
                simulator, session_data,
                inference, x_mean, x_std,
                window_size=window_size,
                stride=stride,
                num_samples=num_samples
            )
            
            all_results[mouse_id][session_id] = results
            session_results.append(results)
            session_true_params.append(true_params)
            
            # Plot individual session
            if analysis_dir:
                save_path = os.path.join(analysis_dir, 
                                        f'mouse{mouse_id}_session{session_id}_evolution.png')
            else:
                save_path = None
            
            plot_parameter_evolution(results, true_params, 
                                   session_id=session_id, mouse_id=mouse_id,
                                   save_path=save_path)
        
        # Plot all sessions for this mouse
        if analysis_dir:
            save_path = os.path.join(analysis_dir, f'mouse{mouse_id}_all_sessions.png')
        else:
            save_path = None
        
        plot_multi_session_comparison(session_results, session_true_params,
                                     mouse_id=mouse_id, save_path=save_path)
    
    print(f"\n{'='*60}")
    print("Dataset analysis complete!")
    if analysis_dir:
        print(f"All plots saved to: {analysis_dir}")
    print(f"{'='*60}")
    
    return all_results


# Test module
if __name__ == "__main__":
    print("Testing sliding window inference module...")
    
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM, create_prior
    from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_inference import train_snle
    
    # Setup
    simulator = PatchForagingDDM()
    prior = create_prior()
    
    print("\n1. Testing data simulation...")
    dataset = simulate_full_dataset(
        simulator, 
        num_mice=2, 
        num_sessions=2,
        sites_per_session=(100, 150),  # Shorter for testing
        drift_increase=0.2
    )
    print(f"   Dataset created: {len(dataset['mice'])} mice")
    
    print("\n2. Training quick SNLE model...")
    _, inference, x_mean, x_std, _ = train_snle(
        simulator, prior,
        num_simulations=1000,
        window_sites=100,
        mode='multi',
        max_num_epochs=5
    )
    print("   Model trained")
    
    print("\n3. Testing sliding window inference...")
    mouse_id = 0
    session_id = 0
    session_data, true_params = dataset['data'][mouse_id][session_id]
    
    results = sliding_window_inference(
        simulator, session_data,
        inference, x_mean, x_std,
        window_size=50,
        stride=10,
        num_samples=100,
        warmup_steps=50
    )
    print(f"   Analyzed {results['num_windows']} windows")
    
    print("\n4. Testing visualization...")
    plot_parameter_evolution(results, true_params, 
                           session_id=session_id, mouse_id=mouse_id)
    print("   Plot generated")
    
    print("\n✓ All tests passed!")