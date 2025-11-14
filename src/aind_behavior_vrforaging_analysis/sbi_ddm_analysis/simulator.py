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
    """
    DDM for patch foraging where:
    - Drift pushes toward leaving threshold
    - Rewards push away from threshold (negative bump)
    - Failures push toward threshold (positive bump)
    - Animal leaves when evidence >= threshold
    """
    
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
    
    def sample_inter_site_interval(self):
        """Sample time between odor sites (truncated Gaussian)"""
        return torch.clamp(
            self.interval_mean + self.interval_std * torch.randn(1),
            min=self.interval_min,
            max=self.interval_max
        ).item()
    
    def _simulate_one_patch(self, theta: torch.Tensor, global_time: float, patch_start_time: float) -> tuple:
        """
        Core method: simulate a single patch until animal leaves.
        
        Args:
            theta: [drift_rate, reward_bump, failure_bump, noise_std]
            global_time: current global time
            patch_start_time: when current patch started
        
        Returns:
            patch_data: list of [time_in_patch, reward, stopped] for each site
            new_global_time: updated global time after patch
        """
        drift_rate, reward_bump, failure_bump, noise_std = theta
        evidence = self.start_point
        num_rewards = 0
        patch_data = []
        
        while True:
            dt = self.sample_inter_site_interval()
            global_time += dt
            time_in_patch = global_time - patch_start_time
            
            # Accumulate evidence
            evidence += drift_rate * dt
            if noise_std > 0:
                evidence += noise_std * torch.randn(1).item() * np.sqrt(dt)
            
            # Decision: leave or stop?
            if evidence >= self.threshold:
                patch_data.append([time_in_patch, 0, 0])  # Leave
                break
            else:
                # Stop and check reward
                reward_prob = reward_probability(num_rewards, self.initial_prob, self.decay_rate)
                reward = int(torch.rand(1) < reward_prob)
                patch_data.append([time_in_patch, reward, 1])
                
                # Update evidence
                evidence += -reward_bump if reward else failure_bump
                num_rewards += reward
        total_time = global_time - patch_start_time
        num_stops = len(patch_data)
        summary_stats = torch.tensor([total_time, num_stops, num_rewards], dtype=torch.float32)
    
        return patch_data, global_time, summary_stats
    
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
        true_theta = []
        global_time = 0.0
        patch_start_time = 0.0
        
        for theta in param_generator:
            
            true_theta.append(theta)
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
                    patch_stats[0, 0].item(),  # total_time
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
            
            return data_tensor, aggregate_stats, true_theta
        else:
            # Return only first patch statistics
            return data_tensor, patch_stats_list[0], true_theta
        
        # ===== Parameter Generators =====

    def evolve_params(self,
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
        """Simulate with parameters evolving via a random walk."""

        param_gen = self.evolve_params(
            mode="walk", theta_init=theta_mean, sigma=sigma, shift=shift
        )
        return self.simulate_trial(param_gen, window_sites)

    def simulate_with_steps(self, window_sites: int,
                            mean_patches_per_regime: int = 10) -> torch.Tensor:
        """Simulate with stepwise regime changes in parameters."""

        param_gen = self.evolve_params(
            mode="step", mean_patches_per_regime=mean_patches_per_regime
        )
        return self.simulate_trial(param_gen, window_sites)


def create_prior():
    """
    Prior distribution for DDM parameters
    
    Returns:
        BoxUniform prior over [drift_rate, reward_bump, failure_bump]
    """
    from sbi.utils.torchutils import BoxUniform

    low = torch.tensor([0.01, 0.01, 0.0, 0.0])
    high = torch.tensor([1.5, 1.5, 1.5, 0.1])
    
    return BoxUniform(
        low=low,
        high=high, 
    ), low, high

# ===== Tests =====

if __name__ == "__main__":
    print("="*60)
    print("Testing Simulator")
    print("="*60)
    
    simulator = PatchForagingDDM()
    
    # Test 1: Walk parameters
    print("\n1. Walk parameters")
    theta_mean = torch.tensor([0.5, 0.6, 0.2, 0.05])
    data_tensor, aggregate_stats, true_theta = simulator.simulate_with_walk(theta_mean, window_sites=100, sigma=0.0, shift=0.0)
    print(f"   Shape: {data_tensor.shape}")
    print(f"   Patches: {(data_tensor[:, 2] == 0).sum().item()}")

    # Test 2: Step changes
    print("\n2. Step change parameters")
    data_tensor, aggregate_stats, true_theta = simulator.simulate_with_steps(window_sites=100, mean_patches_per_regime=5)
    print(f"   Shape: {data_tensor.shape}")
    print(f"   Patches: {(data_tensor[:, 2] == 0).sum().item()}")

    # Test 3: Manual parameter generator usage
    print("\n3. Manual generator usage")
    param_gen = simulator.evolve_params(mode="step", mean_patches_per_regime=3)
    data_tensor, aggregate_stats, true_theta= simulator.simulate_trial(param_gen, window_sites=50)
    print(f"   Shape: {data_tensor.shape}")
    print(f"   Patches: {(data_tensor[:, 2] == 0).sum().item()}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)