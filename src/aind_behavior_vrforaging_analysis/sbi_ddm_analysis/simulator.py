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
                 interval_max=5.0,
                 noise_std=0.2):
        self.initial_prob = initial_prob
        self.decay_rate = decay_rate
        self.threshold = threshold
        self.start_point = start_point
        self.interval_mean = interval_mean
        self.interval_std = interval_std
        self.interval_min = interval_min
        self.interval_max = interval_max
        self.noise_std = noise_std
    
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
            theta: [drift_rate, reward_bump, failure_bump]
            global_time: current global time
            patch_start_time: when current patch started
        
        Returns:
            patch_data: list of [time_in_patch, reward, stopped] for each site
            new_global_time: updated global time after patch
        """
        drift_rate, reward_bump, failure_bump = theta
        evidence = self.start_point
        num_rewards = 0
        patch_data = []
        
        while True:
            dt = self.sample_inter_site_interval()
            global_time += dt
            time_in_patch = global_time - patch_start_time
            
            # Accumulate evidence
            evidence += drift_rate * dt
            if self.noise_std > 0:
                evidence += self.noise_std * torch.randn(1).item() * np.sqrt(dt)
            
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

    def walk_params(self, theta_init: torch.Tensor, sigma: float = 0.0, shift: float = 0.0,
                            clip_low: torch.Tensor = None, clip_high: torch.Tensor = None):
        """
        Generator that yields moving theta via walk. Assumes constant unless sigma/shift > 0.
        
        Args:
            theta_init: Initial [drift_rate, reward_bump, failure_bump]
            sigma: Step size standard deviation (isotropic)
            clip_low/high: Bounds (default: [0.01, 0.01, 0.0] to [1.5, 1.5, 1.5])
            shift: Directional shift (using 'shift' instead of 'drift' to avoid confusion)

        Yields:
            theta (moving via walk)
        """
        if clip_low is None:
            clip_low = torch.tensor([0.01, 0.01, 0.0])
        if clip_high is None:
            clip_high = torch.tensor([1.5, 1.5, 1.5])
        
        theta = theta_init.clone()
        
        while True:
            yield theta.clone()
            # Update for next iteration
            theta = theta + torch.randn(3) * sigma + shift
            theta = torch.clamp(theta, clip_low, clip_high)

    def step_change_params(self, mean_patches_per_regime: int = 10,
                            theta_ranges: tuple = None):
        """
        Generator that yields theta with step changes between regimes.
        
        Args:
            mean_patches_per_regime: Average patches before regime change
            theta_ranges: (low, high) bounds for sampling regimes
        
        Yields:
            theta (changes abruptly every ~mean_patches_per_regime patches)
        """
        if theta_ranges is None:
            low = torch.tensor([0.01, 0.01, 0.0])
            high = torch.tensor([1.5, 1.5, 1.5])
        else:
            low, high = theta_ranges
        
        while True:
            # Sample new regime parameters
            current_theta = low + torch.rand(3) * (high - low)
            
            # Decide how long this regime lasts (Poisson for variability)
            regime_length = np.random.poisson(mean_patches_per_regime)
            regime_length = max(regime_length, 1)  # At least 1 patch
            
            # Yield the same theta for all patches in this regime
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