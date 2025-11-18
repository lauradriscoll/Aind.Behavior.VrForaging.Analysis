"""
Simple SNLE Parameter Sweep (JAX version)
-----------------------------------------
Evaluates how training size (num_simulations) affects posterior accuracy.

Uses: PatchForagingDDM_JAX simulator
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import pandas as pd
from datetime import datetime
import logging
import jax.numpy as jnp
from jax import random

from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator import PatchForagingDDM_JAX, create_prior
from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.snle.snle_inference_jax import train_snle, infer_parameters_snle


# --- Sweep parameters ---
NUM_SIMULATIONS = [10000, 50000, 100000]

TEST_CASES = [
    ("low_drift", jnp.array([0.2, 0.8, 0.3, 0.01])),
    ("high_drift", jnp.array([0.8, 0.2, 0.3, 0.01])),
    ("balanced", jnp.array([0.5, 0.5, 0.5, 0.01])),
]


# --- Helpers ---
def setup_logger(results_dir):
    """Create logger that writes to both file and console."""
    log_file = os.path.join(results_dir, "sweep_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("sweep")


def evaluate_case(snle, snle_params, simulator, true_theta, y_mean, y_std, rng_key):
    """
    Evaluate SNLE on a single test case.
    
    Args:
        snle: Trained SNLE model
        snle_params: Trained parameters
        simulator: PatchForagingDDM_JAX instance
        true_theta: True parameter values (4,)
        y_mean, y_std: Normalization statistics
        rng_key: JAX random key
        
    Returns:
        dict with evaluation metrics
    """
    # Generate one observed dataset from true_theta
    rng_key, obs_key = random.split(rng_key)
    _, observed_stats = simulator.simulate_one_window(true_theta, obs_key)
    
    # Infer posterior
    rng_key, infer_key = random.split(rng_key)
    posterior_samples, _ = infer_parameters_snle(
        snle=snle,
        snle_params=snle_params,
        observed_stats=observed_stats,
        y_mean=y_mean,
        y_std=y_std,
        num_samples=1000,
        num_warmup=200,
        num_chains=4,
        rng_key=infer_key
    )
    
    # Compute MAE (mean absolute error)
    posterior_mean = posterior_samples.mean(axis=0)
    mae = jnp.abs(posterior_mean - true_theta).mean()
    
    # Compute coverage (does true_theta fall within 95% credible interval?)
    lower = jnp.percentile(posterior_samples, 2.5, axis=0)
    upper = jnp.percentile(posterior_samples, 97.5, axis=0)
    coverage = jnp.mean((true_theta >= lower) & (true_theta <= upper))
    
    return {
        "mae": float(mae),
        "coverage": float(coverage),
        "posterior_mean": posterior_mean.tolist(),
        "posterior_std": posterior_samples.std(axis=0).tolist(),
    }


def train_and_eval(num_sims, results_dir, rng_key, logger):
    """
    Train SNLE and evaluate on test cases.
    
    Args:
        num_sims: Number of simulations for training
        results_dir: Directory to save results
        rng_key: JAX random key
        logger: Logger instance
        
    Returns:
        DataFrame with results for all test cases
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training SNLE: {num_sims} simulations")
    logger.info(f"{'='*60}")

    # Initialize simulator and prior
    simulator = PatchForagingDDM_JAX()
    prior_fn = create_prior(
        prior_low=jnp.array([0.0, 0.0, 0.0, 0.0]),
        prior_high=jnp.array([1.0, 1.0, 1.0, 0.01])
    )
    
    # Train SNLE
    rng_key, train_key = random.split(rng_key)
    snle, snle_params, losses, _, y_mean, y_std = train_snle(
        simulator=simulator,
        prior_fn=prior_fn,
        mode='multi',
        n_simulations=int(num_sims),
        rng_key=train_key,
    )
    
    logger.info(f"Training complete. Final loss: {losses}")
    
    # Evaluate on test cases
    results = []
    for case_name, true_theta in TEST_CASES:
        logger.info(f"\nEvaluating case: {case_name}")
        logger.info(f"True theta: {true_theta}")
        
        rng_key, eval_key = random.split(rng_key)
        metrics = evaluate_case(
            snle=snle,
            snle_params=snle_params,
            simulator=simulator,
            true_theta=true_theta,
            y_mean=y_mean,
            y_std=y_std,
            rng_key=eval_key
        )
        
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  Coverage: {metrics['coverage']:.2f}")
        logger.info(f"  Posterior mean: {metrics['posterior_mean']}")
        
        metrics["case"] = case_name
        metrics["true_theta"] = true_theta.tolist()
        results.append(metrics)
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    df["num_simulations"] = num_sims
    
    # Append to summary CSV
    csv_path = os.path.join(results_dir, "sweep_summary.csv")
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=False)
    
    logger.info(f"\nResults saved to {csv_path}")
    
    return df


# --- Main sweep ---
def run_sweep(base_dir="snle_sweep_jax"):
    """
    Run parameter sweep over num_simulations.
    
    Args:
        base_dir: Base directory for results
        
    Returns:
        results_dir: Path to results directory
        all_results: Combined DataFrame with all results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, f"sweep_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    logger = setup_logger(results_dir)
    logger.info(f"Starting parameter sweep")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Sweep parameters: {NUM_SIMULATIONS}")
    logger.info(f"Test cases: {[name for name, _ in TEST_CASES]}")

    rng_key = random.PRNGKey(0)
    all_results = []

    for num_sims in NUM_SIMULATIONS:
        try:
            rng_key, sweep_key = random.split(rng_key)
            df = train_and_eval(num_sims, results_dir, sweep_key, logger)
            all_results.append(df)
        except Exception as e:
            logger.error(f"Failed for num_sims={num_sims}: {e}", exc_info=True)

    logger.info("\n" + "="*60)
    logger.info("Sweep complete!")
    logger.info("="*60)
    
    final_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    return results_dir, final_df


if __name__ == "__main__":
    results_dir, results_df = run_sweep()
    print(f"\nResults saved to: {results_dir}")
    print("\nSummary:")
    print(results_df[["num_simulations", "case", "mae", "coverage"]])