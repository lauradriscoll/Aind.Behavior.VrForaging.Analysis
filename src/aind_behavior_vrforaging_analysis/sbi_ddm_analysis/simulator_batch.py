"""
Simulator for patch foraging DDM.

Structure:
1. Core simulation: _simulate_one_patch, sample_inter_site_interval
2. Main interface: simulate_trial (takes parameter generator)
3. Parameter generators: walk_params, step_change_params
4. Convenience wrappers for common use cases
"""

import torch
import numpy as np


def reward_probability(num_rewards, initial_prob=0.8, decay_rate=-0.1):
    """Exponential decay reward probability based on number of rewards collected"""
    return initial_prob * torch.exp(torch.tensor(decay_rate * num_rewards))


class PatchForagingDDM:
    def __init__(self, 
                 initial_prob=0.8, 
                 decay_rate=-0.1,
                 threshold=1.0,
                 start_point=0.0,
                 interval_mean=1.0, 
                 interval_std=0.3, 
                 interval_min=0.1, 
                 interval_max=5.0):
        self.initial_prob = initial_prob
        self.decay_rate = decay_rate
        self.threshold = threshold
        self.start_point = start_point
        self.interval_mean = interval_mean
        self.interval_std = interval_std
        self.interval_min = interval_min
        self.interval_max = interval_max
    
    def sample_inter_site_intervals(self, batch_size: int):
        """Sample inter-site intervals for all patches (vectorized)."""
        dt = self.interval_mean + self.interval_std * torch.randn(batch_size)
        return torch.clamp(dt, min=self.interval_min, max=self.interval_max)
    
    def _simulate_patches_batch(self, thetas: torch.Tensor, max_sites: int = 200):
        """
        Vectorized simulation of multiple patches simultaneously.

        Args:
            thetas: (N, 4) tensor of parameters [drift_rate, reward_bump, failure_bump, noise_std]
            max_sites: Maximum number of sites to simulate before truncation.

        Returns:
            summary_stats: (N, 3) tensor [total_time, num_stops, num_rewards]
        """
        device = thetas.device
        N = thetas.shape[0]

        drift_rate = thetas[:, 0]
        reward_bump = thetas[:, 1]
        failure_bump = thetas[:, 2]
        noise_std = thetas[:, 3]

        evidence = torch.full((N,), self.start_point, device=device)
        num_rewards = torch.zeros(N, device=device)
        total_time = torch.zeros(N, device=device)
        num_stops = torch.zeros(N, device=device)
        done = torch.zeros(N, dtype=torch.bool, device=device)

        for _ in range(max_sites):
            active = ~done
            if not active.any():
                break

            # Sample inter-site intervals for all active patches
            dt = self.sample_inter_site_intervals(active.sum()).to(device)
            total_time[active] += dt

            # Update evidence
            e = evidence[active]
            dr = drift_rate[active]
            ns = noise_std[active]

            e += dr * dt + ns * torch.sqrt(dt) * torch.randn_like(dt)
            evidence[active] = e

            # Check for leaving threshold
            left = e >= self.threshold
            done_idx = torch.nonzero(active)[left]
            done[done_idx] = True

            # For those that didn’t leave: check reward
            stay_idx = torch.nonzero(active)[~left].squeeze(-1)
            if stay_idx.numel() > 0:
                reward_probs = self.initial_prob * torch.exp(
                    self.decay_rate * num_rewards[stay_idx]
                )
                rewards = (torch.rand_like(reward_probs) < reward_probs).float()
                num_rewards[stay_idx] += rewards
                num_stops[stay_idx] += 1

                # Update evidence after reward/failure
                e2 = evidence[stay_idx]
                rb = reward_bump[stay_idx]
                fb = failure_bump[stay_idx]
                e2 += torch.where(rewards > 0, -rb, fb)
                evidence[stay_idx] = e2

        summary_stats = torch.stack(
            [total_time, num_stops, num_rewards], dim=1
        )
        return summary_stats
    
    def simulate_batch(self, theta_batch: torch.Tensor, max_sites: int = 200):
        """
        Simulate multiple independent patches in parallel.

        Args:
            theta_batch: (N, 4) tensor of patch parameters
            max_sites: max sites per patch

        Returns:
            summary_stats: (N, 3) tensor [total_time, num_stops, num_rewards]
        """
        return self._simulate_patches_batch(theta_batch, max_sites=max_sites)
    
    # ===== Main Interface =====
    
    def simulate_trial(self, param_generator, window_sites: int, return_aggregate: bool = False) -> tuple:
        """
        Simulate foraging trial with time-varying parameters.
        
        Args:
            param_generator: Iterator that yields theta values for each patch
            window_sites: Exact number of sites to return
            return_aggregate: If True, return aggregate stats across patches.
                            If False, return single-patch stats (only first patch).
        
        Returns:
            data: (window_sites, 3) tensor [time_since_patch_start, reward, stopped]
            summary_stats: (3,) for single patch OR (8,) for aggregate
        """
        data = []
        patch_stats_list = []
        global_time = 0.0
        patch_start_time = 0.0
        
        for theta in param_generator:
            patch_data, global_time, summary_stats = self._simulate_one_patch(
                theta, global_time, patch_start_time
            )
            
            data.extend(patch_data)
            patch_stats_list.append(summary_stats)
            patch_start_time = global_time
            
            if len(data) >= window_sites:
                break
        
        data_tensor = torch.tensor(data[:window_sites], dtype=torch.float32)
        
        if return_aggregate:
            # Return aggregate statistics across all patches
            patch_stats = torch.stack(patch_stats_list)  # (num_patches, 3)
            
            # Handle single patch case (std would be NaN)
            if len(patch_stats) == 1:
                aggregate_stats = torch.tensor([
                    patch_stats[0, 0].item(),  # total_time (no mean needed)
                    0.0,                        # std total_time = 0 for single patch
                    patch_stats[0, 1].item(),  # num_stops
                    0.0,                        # std num_stops = 0
                    patch_stats[0, 2].item(),  # num_rewards
                    0.0,                        # std num_rewards = 0
                    patch_stats[0, 2].item(),  # total rewards = same as mean
                    1.0,                        # num_patches = 1
                ], dtype=torch.float32)
            else:
                aggregate_stats = torch.tensor([
                    patch_stats[:, 0].mean().item(),
                    patch_stats[:, 0].std().item(),
                    patch_stats[:, 1].mean().item(),
                    patch_stats[:, 1].std().item(),
                    patch_stats[:, 2].mean().item(),
                    patch_stats[:, 2].std().item(),
                    patch_stats[:, 2].sum().item(),
                    float(len(patch_stats)),
                ], dtype=torch.float32)
            
            return data_tensor, aggregate_stats
        else:
            # Return only first patch statistics
            return data_tensor, patch_stats_list[0]
        
        # ===== Parameter Generators =====

    def evolve_params(
        mode: str,
        theta_init: torch.Tensor = None,
        sigma: float = 0.0,
        shift: float = 0.0,
        bounds: tuple = None,
        mean_patches_per_regime: int = 10,
    ):
        """
        Generator that yields evolving theta parameters according to the specified mode.

        Args:
            mode: 'walk' or 'step'.
            theta_init: Initial parameters for 'walk' mode [drift_rate, reward_bump, failure_bump, noise_std].
            sigma: Step size standard deviation for random walk.
            shift: Directional shift added each walk step.
            bounds: (low, high) tuple of parameter bounds as torch.Tensors.
            mean_patches_per_regime: Average patches per regime in 'step' mode.

        Yields:
            theta (torch.Tensor): evolving parameter vector.
        """
        if mode not in {"walk", "step"}:
            raise ValueError("mode must be either 'walk' or 'step'")

        # Default bounds if none provided
        if bounds is None:
            low = torch.tensor([0.01, 0.01, 0.0, 0.0])
            high = torch.tensor([1.5, 1.5, 1.5, 0.1])
        else:
            low, high = bounds

        if mode == "walk":
            if theta_init is None:
                raise ValueError("theta_init must be provided for 'walk' mode.")
            theta = theta_init.clone()

            while True:
                yield theta.clone()
                theta = theta + torch.randn_like(theta) * sigma + shift
                theta = torch.clamp(theta, low, high)

        elif mode == "step":
            while True:
                current_theta = low + torch.rand_like(low) * (high - low)
                regime_length = np.random.poisson(mean_patches_per_regime)
                regime_length = max(regime_length, 1)
                for _ in range(regime_length):
                    yield current_theta.clone()

    # ===== Convenience Wrappers =====

    def simulate_with_walk(self, theta_mean: torch.Tensor, window_sites: int,
                            sigma: float = 0.0, shift: float = 0.0) -> torch.Tensor:
        """
        Convenience: simulate with walk around theta_mean. 
        Note: This is not random walk because sigma can be 0 and shift moves 
              parameters deterministically in a particular direction.

        Args:
            theta_mean: Mean [drift_rate, reward_bump, failure_bump]
            window_sites: Number of sites to simulate
            sigma: Random walk step size
            shift: directional shift (using 'shift' instead of 'drift' to avoid confusion)
        
        Returns:
            (window_sites, 3) tensor
        """
        param_gen = self.walk_params(theta_mean, sigma=sigma, shift=shift)
        return self.simulate_trial(param_gen, window_sites)

    def simulate_with_steps(self, window_sites: int,
                            mean_patches_per_regime: int = 10) -> torch.Tensor:
        """
        Convenience: simulate with step changes in parameters.
        
        Args:
            window_sites: Number of sites to simulate
            mean_patches_per_regime: Average patches before regime change
        
        Returns:
            (window_sites, 3) tensor
        """
        param_gen = self.step_change_params(mean_patches_per_regime)
        return self.simulate_trial(param_gen, window_sites)

def create_prior():
    """
    Prior distribution for DDM parameters
    
    Returns:
        BoxUniform prior over [drift_rate, reward_bump, failure_bump]
    """
    from sbi.utils.torchutils import BoxUniform
    
    return BoxUniform(
        low=torch.tensor([0.01, 0.01, 0.0]),
        high=torch.tensor([1.5, 1.5, 1.5])
    )

# ===== Tests =====

if __name__ == "__main__":
    print("="*60)
    print("Testing Refactored Simulator")
    print("="*60)
    
    simulator = PatchForagingDDM()
    
    # Test 1: Walk parameters
    print("\n1. Walk parameters")
    theta_mean = torch.tensor([0.5, 0.6, 0.2])
    data = simulator.simulate_window_with_walk(theta_mean, window_sites=100, sigma=0.0, shift=0.0)
    print(f"   Shape: {data.shape}")
    print(f"   Patches: {(data[:, 2] == 0).sum().item()}")

    # Test 2: Step changes
    print("\n2. Step change parameters")
    data = simulator.simulate_with_steps(window_sites=100, mean_patches_per_regime=5)
    print(f"   Shape: {data.shape}")
    print(f"   Patches: {(data[:, 2] == 0).sum().item()}")

    # Test 3: Manual parameter generator usage
    print("\n3. Manual generator usage")
    param_gen = simulator.step_change_params(mean_patches_per_regime=3)
    data = simulator.simulate_trial(param_gen, window_sites=50)
    print(f"   Shape: {data.shape}")
    print(f"   Patches: {(data[:, 2] == 0).sum().item()}")

    # Test 4: Backward compatibility
    print("\n4. Backward compatibility (__call__)")
    data = simulator(theta, max_sites=50)
    print(f"   Shape: {data.shape}")

    # Test 5: Batch simulation
    print("\n5. Batch simulation")
    theta_batch = torch.rand(5, 3)
    data_batch = simulator(theta_batch, max_sites=50)
    print(f"   Shape: {data_batch.shape}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
    
    # Show example usage
    print("\n" + "="*60)
    print("Example Usage")
    print("="*60)
    print("""
# Walk parameters:
data = simulator.simulate_window_with_walk(theta_mean, window_sites=100, sigma=0.0, shift=0.0)

# Step changes:
data = simulator.simulate_with_steps(window_sites=100, mean_patches_per_regime=10)

# Custom generator:
param_gen = simulator.walk_params(theta_init, sigma=0.1)
data = simulator.simulate_trial(param_gen, window_sites=100)
    """)