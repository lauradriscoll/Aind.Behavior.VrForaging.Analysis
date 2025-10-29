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
                 noise_std=0.0):
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
    
    def simulate_trial(self, theta: torch.Tensor, max_sites: int) -> torch.Tensor:
        """
        Simulate patch foraging until max_sites odor sites are encountered.
        May span multiple patches.
        
        Args:
            theta: [drift_rate, reward_bump, failure_bump]
                - drift_rate: positive value, drift toward leaving
                - reward_bump: positive value, pushes away from threshold (evidence -= reward_bump)
                - failure_bump: positive value, pushes toward threshold (evidence += failure_bump)
            max_sites: number of odor sites to simulate
            
        Returns:
            torch.Tensor of shape (max_sites, 3):
            - x1: time_since_patch_start
            - x2: reward_obtained (0 or 1)
            - x3: stopped (1) or skipped/left (0)
        """
        drift_rate, reward_bump, failure_bump = theta
        
        data = []
        patch_start_time = 0.0
        global_time = 0.0
        
        while len(data) < max_sites:
            # Start new patch
            evidence = self.start_point
            site_num = 0
            num_rewards_in_patch = 0
            
            # Simulate within patch until leaving
            while len(data) < max_sites:
                site_num += 1
                
                # Time to next odor site
                dt = self.sample_inter_site_interval()
                global_time += dt
                time_in_patch = global_time - patch_start_time
                
                # Accumulate drift toward leaving + noise
                evidence += drift_rate * dt
                if self.noise_std > 0:
                    evidence += self.noise_std * torch.randn(1).item() * np.sqrt(dt)
                
                # Decision point: check if evidence crosses threshold
                if evidence >= self.threshold:
                    # Skip site, leave patch
                    data.append([time_in_patch, 0, 0])
                    
                    # Reset for next patch
                    patch_start_time = global_time
                    break
                else:
                    # Stop at site, check for reward
                    reward_prob = reward_probability(
                        num_rewards_in_patch, 
                        self.initial_prob, 
                        self.decay_rate
                    )
                    reward = int(torch.rand(1) < reward_prob)
                    
                    data.append([time_in_patch, reward, 1])
                    
                    # Update evidence based on outcome
                    if reward == 1:
                        num_rewards_in_patch += 1
                        evidence -= reward_bump  # Push away from leaving
                    else:
                        evidence += failure_bump  # Push toward leaving
        
        return torch.tensor(data[:max_sites], dtype=torch.float32)
    
    def __call__(self, theta: torch.Tensor, max_sites: int = 50) -> torch.Tensor:
        """Allow calling simulator as a function"""
        if theta.dim() == 1:
            return self.simulate_trial(theta, max_sites)
        else:
            # Batch simulation
            return torch.stack([
                self.simulate_trial(theta[i], max_sites) 
                for i in range(theta.shape[0])
            ])


def create_prior():
    """
    Prior distribution for DDM parameters
    
    Returns:
        BoxUniform prior over [drift_rate, reward_bump, failure_bump]
    """
    from sbi.utils.torchutils import BoxUniform
    
    return BoxUniform(
        low=torch.tensor([0.01, 0.01, 0.0]),    # drift, reward_bump, failure_bump min
        high=torch.tensor([2.0, 2.0, 2.0])       # drift, reward_bump, failure_bump max
    )


# Test the simulator
if __name__ == "__main__":
    simulator = PatchForagingDDM()
    
    # Test parameters: [drift_rate, reward_bump, failure_bump]
    theta = torch.tensor([0.3, 0.4, 0.15])
    
    # Simulate trial with 50 odor sites
    data = simulator(theta, max_sites=50)
    
    print("Data shape:", data.shape)  # Should be (50, 3)
    print("\nFirst 20 sites:")
    print("time | reward | stopped")
    print(data[:20])
    
    # Analyze patch structure
    stopped = data[:, 2]
    left_sites = (stopped == 0).nonzero(as_tuple=True)[0]
    print(f"\nPatches ended at sites: {left_sites.tolist()}")
    print(f"Number of patches: {len(left_sites)}")
    
    # Count rewards
    rewards = data[:, 1]
    total_rewards = rewards.sum().item()
    reward_rate = total_rewards / max_sites
    print(f"\nTotal rewards: {total_rewards}")
    print(f"Reward rate: {reward_rate:.2%}")