import torch
import numpy as np

def reward_probability(num_rewards, initial_prob, decay_rate):
    """Exponential decay reward probability based on number of rewards collected"""
    return initial_prob * torch.exp(torch.tensor(decay_rate * num_rewards))

class DDMSimulator:
    """
    DDM for patch foraging with observable offer acceptance decisions
    
    Evidence accumulates with drift until threshold (normalized to 1.0).
    Sites probabilistically contain rewards.
    When reward is obtained, evidence is pushed DOWN (away from threshold).
    Agent visits sites until threshold is reached, then leaves.
    """
    
    def __init__(self, initial_prob=0.8, decay_rate=-0.1, max_time=50.0, noise_std=0.0,
                 interval_mean=1.0, interval_std=0.3, interval_min=0.1, interval_max=5.0):
        self.initial_prob = initial_prob
        self.decay_rate = decay_rate
        self.max_time = max_time
        self.noise_std = noise_std
        self.interval_mean = interval_mean
        self.interval_std = interval_std
        self.interval_min = interval_min
        self.interval_max = interval_max
        self.threshold = 1.0  # Normalized threshold
    
    def simulate_single(self, theta: torch.Tensor, save_trace=False):
        """
        Args:
            theta: [drift, reward_bump] 
                   drift: rate of evidence accumulation toward threshold
                   reward_bump: amount evidence decreases when reward obtained
            save_trace: If True, return full trajectory
            
        Returns:
            dict with:
                'offer_times': when sites were visited
                'rewards': 1=site had reward, 0=site was empty
            or full trace if save_trace=True
        """
        drift, reward_bump = theta
        
        evidence = 0.0
        time = 0.0
        num_rewards_collected = 0
        
        offer_times = []
        rewards = []    # 1 = site had reward, 0 = site was empty
        
        if save_trace:
            trace = {
                'times': [0.0],
                'evidence': [0.0],
                'offer_times': [],
                'rewards': [],
                'rewarded_times': [],
                'empty_times': []
            }
        
        while time < self.max_time:
            # Time to next site (truncated normal)
            dt = torch.clamp(
                self.interval_mean + self.interval_std * torch.randn(1),
                min=self.interval_min, 
                max=self.interval_max
            ).item()
            
            time += dt
            
            if time >= self.max_time:
                break
            
            # Accumulate evidence with drift (and optional noise)
            evidence += drift * dt
            if self.noise_std > 0:
                evidence += self.noise_std * torch.randn(1).item() * np.sqrt(dt)
            
            if save_trace:
                trace['times'].append(time)
                trace['evidence'].append(evidence)
            
            # Check threshold - if reached, leave without checking this site
            if evidence >= self.threshold:
                break
            
            # Agent checks site for reward
            offer_times.append(time)
            
            # Check if site has reward (probabilistic)
            prob = reward_probability(num_rewards_collected, self.initial_prob, self.decay_rate)
            has_reward = (torch.rand(1) < prob).item()
            rewards.append(int(has_reward))
            
            if has_reward:
                num_rewards_collected += 1
                evidence -= reward_bump  # Evidence pushed back down
                
                if save_trace:
                    trace['rewarded_times'].append(time)
            else:
                if save_trace:
                    trace['empty_times'].append(time)
            
            if save_trace:
                trace['offer_times'].append(time)
                trace['rewards'].append(int(has_reward))
                trace['times'].append(time)
                trace['evidence'].append(evidence)
            
            # Check threshold again after potential reward bump
            if evidence >= self.threshold:
                break
        
        if save_trace:
            trace['final_time'] = time
            trace['total_rewards'] = num_rewards_collected
            trace['total_sites'] = len(offer_times)
            return trace
        
        return {
            'offer_times': torch.tensor(offer_times, dtype=torch.float32),
            'rewards': torch.tensor(rewards, dtype=torch.float32)
        }
    
    def simulate_batch(self, theta: torch.Tensor, n_trials_per_param=10):
        """
        Simulate multiple trials for each parameter setting
        
        Args:
            theta: [batch_size, 2] or [2] - parameters [drift, reward_bump]
            n_trials_per_param: number of trials to simulate per parameter setting
            
        Returns:
            list of trial dicts (if single param) or list of lists (if batch)
        """
        if theta.dim() == 1:
            # Single parameter setting - simulate n_trials
            trials = []
            for _ in range(n_trials_per_param):
                trials.append(self.simulate_single(theta))
            return trials
        else:
            # Batch of parameter settings
            all_trials = []
            for i in range(theta.shape[0]):
                batch_trials = []
                for _ in range(n_trials_per_param):
                    batch_trials.append(self.simulate_single(theta[i]))
                all_trials.append(batch_trials)
            return all_trials
    
    def __call__(self, theta: torch.Tensor, n_trials=10):
        """For sbi compatibility"""
        return self.simulate_batch(theta, n_trials_per_param=n_trials)


def create_ddm_prior():
    """Prior for DDM parameters"""
    from sbi.utils.torchutils import BoxUniform
    
    return BoxUniform(
        low=torch.tensor([0.01, 0.01]),   # [drift_min, reward_bump_min]
        high=torch.tensor([0.5, 0.5])     # [drift_max, reward_bump_max]
    )