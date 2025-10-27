import torch
import numpy as np

def reward_probability(num_rewards, initial_prob, decay_rate):
    """Exponential decay reward probability based on number of rewards collected"""
    return initial_prob * torch.exp(torch.tensor(decay_rate * num_rewards))

class DDMSimulator:
    """
    Simple DDM for patch foraging
    
    Evidence accumulates until threshold is reached.
    """
    
    def __init__(self, initial_prob=0.8, decay_rate=-0.1, max_time=50.0, noise_std=0,
                 interval_mean=1.0, interval_std=0.3, interval_min=0.1, interval_max=5.0):
        self.initial_prob = initial_prob
        self.decay_rate = decay_rate
        self.max_time = max_time
        self.noise_std = noise_std
        self.interval_mean = interval_mean
        self.interval_std = interval_std
        self.interval_min = interval_min
        self.interval_max = interval_max
    
    def simulate_single(self, theta: torch.Tensor, save_trace=False) -> torch.Tensor:
        """
        Args:
            theta: [drift, reward_pulse]
            save_trace: If True, save evidence and event traces
            
        Returns:
            [time_in_patch, num_rewards] or trace dict if save_trace=True
        """
        drift, reward_pulse = theta
        
        evidence = 0.0
        time = 0.0
        rewards = 0
        gap = 1.0 # Fixed gap

        if save_trace:
            trace = {
                'times': [0.0],
                'evidence': [0.0],
                'reward_times': [],
                'no_reward_times': [],
            }
        
        while time < self.max_time:
            
            # Time to next site (truncated gaussian)
            dt = torch.clamp(
                self.interval_mean + self.interval_std * torch.randn(1),
                min=self.interval_min, 
                max=self.interval_max
            ).item()
            time += dt
            
            if time >= self.max_time:
                break

            # Update evidence
            evidence += (drift + torch.randn(1).item() * self.noise_std)*dt
            
            if save_trace:
                trace['times'].append(time)
                trace['evidence'].append(evidence.item())

            # Check threshold
            if evidence >= gap:
                break
            
            # Check reward probability based on number of rewards collected
            prob = reward_probability(rewards, self.initial_prob, self.decay_rate)
            if torch.rand(1) < prob:
                rewards += 1
                if save_trace:
                    trace['reward_times'].append(time)
                    evidence += -reward_pulse
                    trace['times'].append(time)
                    trace['evidence'].append(evidence.item())
            else:
                if save_trace:
                    trace['no_reward_times'].append(time)
                    trace['times'].append(time)
                    trace['evidence'].append(evidence.item())
            
            # Check threshold
            if evidence >= gap:
                break
        
        if save_trace:
            trace['final_time'] = time
            trace['total_rewards'] = rewards
            return trace
        
        return torch.tensor([time, rewards], dtype=torch.float32)
    
    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.dim() == 1:
            return self.simulate_single(theta)
        else:
            return torch.stack([self.simulate_single(theta[i]) for i in range(theta.shape[0])])


def create_ddm_prior():
    """Prior for DDM parameters"""
    from sbi.utils.torchutils import BoxUniform
    
    return BoxUniform(
        low=torch.tensor([.01, .01]), #drift, reward_pulse min
        high=torch.tensor([1.0, 1.0]) #drift, reward_pulse max
    )