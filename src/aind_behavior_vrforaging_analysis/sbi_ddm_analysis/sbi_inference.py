import torch
from sbi import simulate_for_sbi
from sbi.inference import MNLE
from sbi.inference.posteriors import MCMCPosteriorParameters
from .simulator import DDMSimulator, create_ddm_prior

class DDMInference:
    """SBI-based inference for DDM parameters"""
    
    def __init__(self):
        self.simulator = DDMSimulator()
        self.prior = create_ddm_prior()
        self.inference_method = None
        self.posterior = None
        self.training_data = None
    
    def generate_training_data(self, num_simulations: int = 5000):
        """Generate training data for MNLE"""
        print(f"Generating {num_simulations} DDM simulations...")
        
        theta, x = simulate_for_sbi(
            simulator=self.simulator,
            proposal=self.prior,
            num_simulations=num_simulations
        )
        
        self.training_data = (theta, x)
        print(f"Training data shapes: θ={theta.shape}, x={x.shape}")
        return theta, x
    
    def train_mnle(self, theta=None, x=None):
        """Train MNLE estimator"""
        if theta is None or x is None:
            if self.training_data is None:
                raise ValueError("No training data available. Run generate_training_data() first.")
            theta, x = self.training_data
        
        print("Training MNLE...")
        self.inference_method = MNLE(prior=self.prior)
        self.inference_method.append_simulations(theta, x)
        density_estimator = self.inference_method.train()
        print("MNLE training completed!")
        return density_estimator
    
    def build_posterior(self, **mcmc_kwargs):
        """Build posterior with MCMC sampling"""
        if self.inference_method is None:
            raise ValueError("Must train MNLE first!")
        
        mcmc_params = MCMCPosteriorParameters(
            method="slice_np_vectorized",
            num_chains=4,
            thin=5,
            warmup_steps=100,
            **mcmc_kwargs
        )
        
        self.posterior = self.inference_method.build_posterior(
            posterior_parameters=mcmc_params
        )
        return self.posterior
    
    def infer_parameters(self, observed_data: torch.Tensor, num_samples: int = 1000):
        """Get posterior samples for observed data"""
        if self.posterior is None:
            raise ValueError("Must build posterior first!")
        
        samples = self.posterior.sample((num_samples,), x=observed_data)
        return samples