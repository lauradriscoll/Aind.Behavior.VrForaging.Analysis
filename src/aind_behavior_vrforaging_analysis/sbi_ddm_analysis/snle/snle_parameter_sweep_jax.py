"""
Simple SNLE Parameter Sweep (JAX version)
-----------------------------------------
Evaluates how training size (num_simulations) and window size (window_sites)
affect posterior accuracy and generative quality.

Uses: PatchForagingDDM_JAX simulator
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import torch
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import logging
from jax import random
from scipy.stats import wasserstein_distance
from scipy.special import kl_div

from aind_behavior_vrforaging_analysis.sbi_ddm_analysis.simulator_jax import (
    PatchForagingDDM_JAX, create_prior_jax
)
from snle_inference import train_snle, infer_parameters_snle


# --- Sweep parameters ---
NUM_SIMULATIONS = [1e5, 5e5]
WINDOW_SITES = [50, 100]

TEST_CASES = [
    ("low_drift", torch.tensor([0.2, 0.8, 0.3, 0.01])),
    ("high_drift", torch.tensor([0.8, 0.2, 0.3, 0.01])),
    ("balanced", torch.tensor([0.5, 0.5, 0.5, 0.01])),
]


# --- Helpers ---
def setup_logger(results_dir):
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


def compute_kl(p, q):
    all_data = np.concatenate([p, q])
    bins = np.histogram_bin_edges(all_data, bins=30)
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)
    p_hist /= (p_hist.sum() + 1e-10)
    q_hist /= (q_hist.sum() + 1e-10)
    return np.sum(kl_div(p_hist + 1e-10, q_hist + 1e-10))


def compute_generative_metrics(simulator, true_theta, posterior_samples, rng_key):
    import jax.numpy as jnp

    n = min(200, len(posterior_samples))
    rng_key, k1, k2 = random.split(rng_key, 3)
    true_batch = jnp.tile(jnp.array(true_theta.numpy()), (n, 1))
    _, true_stats = simulator.simulate_batch(true_batch, k1)
    post_batch = jnp.array(posterior_samples[:n].numpy())
    _, post_stats = simulator.simulate_batch(post_batch, k2)

    true_data = np.array(true_stats)
    post_data = np.array(post_stats)

    feats = ["time", "num_stops", "num_rewards"]
    results = {}
    for i, f in enumerate(feats):
        wd = wasserstein_distance(true_data[:, i], post_data[:, i])
        kl = compute_kl(true_data[:, i], post_data[:, i])
        results[f"wasserstein_{f}"] = wd
        results[f"kl_{f}"] = kl

    results["mean_wasserstein"] = np.mean([results[f"wasserstein_{f}"] for f in feats])
    results["mean_kl"] = np.mean([results[f"kl_{f}"] for f in feats])
    return results


def evaluate_case(inference, simulator, true_theta, x_mean, x_std, rng_key):
    # Generate one observed dataset
    param_gen = simulator.evolve_params("walk", theta_init=true_theta, sigma=0.0)
    _, obs_stats, _ = simulator.simulate_trial(param_gen, window_sites=100)

    # Infer posterior
    posterior = infer_parameters_snle(inference, obs_stats, x_mean, x_std,
                                      num_samples=1000, warmup_steps=200)
    mean = posterior.mean(0)
    mae = (mean - true_theta).abs().mean().item()

    # Compute generative metrics
    gen_metrics = compute_generative_metrics(simulator, true_theta, posterior, rng_key)
    return {"mae": mae, **gen_metrics}


def train_and_eval(num_sims, window_sites, results_dir, rng_key, logger):
    logger.info(f"Training SNLE: {num_sims} sims, {window_sites} sites")

    simulator = PatchForagingDDM_JAX()
    prior = create_prior_jax()
    rng_key, subkey = random.split(rng_key)

    _, inference, x_mean, x_std, hist = train_snle(
        simulator=simulator,
        prior=prior,
        num_simulations=int(num_sims),
        window_sites=window_sites,
        mode="single",
        batch_size=64,
        max_num_epochs=50,
        use_jax=True,
        rng_key=subkey,
    )

    # Evaluate test cases
    results = []
    for name, theta in TEST_CASES:
        logger.info(f"Evaluating {name} | theta={theta.numpy()}")
        res = evaluate_case(inference, simulator, theta, x_mean, x_std, rng_key)
        res["case"] = name
        results.append(res)

    # Save results
    df = pd.DataFrame(results)
    df["num_simulations"] = num_sims
    df["window_sites"] = window_sites
    csv_path = os.path.join(results_dir, "sweep_summary.csv")
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=False)

    return df


# --- Main sweep ---
def run_sweep(base_dir="snle_sweep_simple"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, f"sweep_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    logger = setup_logger(results_dir)

    rng_key = random.PRNGKey(0)
    all_results = []

    for n in NUM_SIMULATIONS:
        for w in WINDOW_SITES:
            try:
                df = train_and_eval(n, w, results_dir, rng_key, logger)
                all_results.append(df)
            except Exception as e:
                logger.error(f"Failed for {n}, {w}: {e}")

    logger.info("Sweep complete.")
    return results_dir, pd.concat(all_results, ignore_index=True)


if __name__ == "__main__":
    run_sweep()
