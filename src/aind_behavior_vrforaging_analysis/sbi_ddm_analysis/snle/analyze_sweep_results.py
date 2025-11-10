"""
Utility for analyzing parameter sweep results and choosing optimal hyperparameters.

Use this after running snle_parameter_sweep.py to determine best settings
for sliding window analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import glob
from typing import Dict, Tuple, List


def analyze_sweep_results(results: Dict) -> Dict:
    """
    Analyze parameter sweep results and recommend optimal settings.
    
    Args:
        results: Output from snle_parameter_sweep.py
        
    Returns:
        recommendations: Dict with optimal hyperparameters
    """
    print("="*80)
    print("PARAMETER SWEEP ANALYSIS")
    print("="*80)
    
    # Extract results
    configs = results['configs']
    param_errors = results['param_errors']
    posterior_stds = results['posterior_stds']
    
    # Find configuration with best performance
    best_idx = np.argmin([e.mean().item() for e in param_errors])
    best_config = configs[best_idx]
    best_error = param_errors[best_idx].mean().item()
    best_std = posterior_stds[best_idx].mean().item()
    
    print("\nBEST CONFIGURATION:")
    print(f"  num_simulations: {best_config['num_simulations']}")
    print(f"  window_sites: {best_config['window_sites']}")
    print(f"  Mean parameter error: {best_error:.4f}")
    print(f"  Mean posterior std: {best_std:.4f}")
    
    # Analyze trade-offs
    print("\n" + "="*80)
    print("TRADE-OFF ANALYSIS")
    print("="*80)
    
    # Group by num_simulations
    sim_values = sorted(list(set([c['num_simulations'] for c in configs])))
    window_values = sorted(list(set([c['window_sites'] for c in configs])))
    
    print(f"\nTested {len(sim_values)} simulation counts × {len(window_values)} window sizes")
    print(f"  Simulations: {sim_values}")
    print(f"  Window sizes: {window_values}")
    
    # Effect of num_simulations
    print("\nEFFECT OF NUM_SIMULATIONS (averaged over window sizes):")
    for n_sim in sim_values:
        errors = [param_errors[i].mean().item() for i, c in enumerate(configs) 
                 if c['num_simulations'] == n_sim]
        avg_error = np.mean(errors)
        std_error = np.std(errors)
        print(f"  {n_sim:6d}: error = {avg_error:.4f} ± {std_error:.4f}")
    
    # Effect of window_sites
    print("\nEFFECT OF WINDOW_SITES (averaged over num_simulations):")
    for w_size in window_values:
        errors = [param_errors[i].mean().item() for i, c in enumerate(configs) 
                 if c['window_sites'] == w_size]
        avg_error = np.mean(errors)
        std_error = np.std(errors)
        print(f"  {w_size:3d} sites: error = {avg_error:.4f} ± {std_error:.4f}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    recommendations = {
        'optimal': best_config,
        'quick_test': None,
        'production': None,
    }
    
    # Quick test: smallest config with reasonable performance
    quick_configs = [c for c in configs if c['num_simulations'] <= 10000]
    if quick_configs:
        quick_idx = np.argmin([param_errors[configs.index(c)].mean().item() 
                              for c in quick_configs])
        recommendations['quick_test'] = quick_configs[quick_idx]
    
    # Production: best config with num_simulations >= 20000
    prod_configs = [c for c in configs if c['num_simulations'] >= 20000]
    if prod_configs:
        prod_idx = np.argmin([param_errors[configs.index(c)].mean().item() 
                             for c in prod_configs])
        recommendations['production'] = prod_configs[prod_idx]
    
    print("\n1. OPTIMAL (best performance):")
    print(f"   num_simulations = {recommendations['optimal']['num_simulations']}")
    print(f"   window_sites = {recommendations['optimal']['window_sites']}")
    
    if recommendations['quick_test']:
        print("\n2. QUICK TEST (fast iteration):")
        print(f"   num_simulations = {recommendations['quick_test']['num_simulations']}")
        print(f"   window_sites = {recommendations['quick_test']['window_sites']}")
    
    if recommendations['production']:
        print("\n3. PRODUCTION (reliable performance):")
        print(f"   num_simulations = {recommendations['production']['num_simulations']}")
        print(f"   window_sites = {recommendations['production']['window_sites']}")
    
    return recommendations


def plot_sweep_heatmap(results: Dict, save_path: str = None):
    """
    Create heatmap showing parameter error across sweep configurations.
    
    Args:
        results: Output from parameter sweep
        save_path: Where to save plot
    """
    configs = results['configs']
    param_errors = results['param_errors']
    
    # Extract unique values
    sim_values = sorted(list(set([c['num_simulations'] for c in configs])))
    window_values = sorted(list(set([c['window_sites'] for c in configs])))
    
    # Create error matrix
    error_matrix = np.zeros((len(window_values), len(sim_values)))
    
    for i, c in enumerate(configs):
        sim_idx = sim_values.index(c['num_simulations'])
        window_idx = window_values.index(c['window_sites'])
        error_matrix[window_idx, sim_idx] = param_errors[i].mean().item()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(error_matrix, cmap='viridis_r', aspect='auto')
    
    # Labels
    ax.set_xticks(range(len(sim_values)))
    ax.set_xticklabels([f'{s//1000}k' for s in sim_values])
    ax.set_xlabel('Number of Training Simulations', fontsize=12)
    
    ax.set_yticks(range(len(window_values)))
    ax.set_yticklabels(window_values)
    ax.set_ylabel('Window Sites', fontsize=12)
    
    ax.set_title('Mean Parameter Error Across Sweep', fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Absolute Error', fontsize=11)
    
    # Annotate cells
    for i in range(len(window_values)):
        for j in range(len(sim_values)):
            text = ax.text(j, i, f'{error_matrix[i, j]:.3f}',
                         ha="center", va="center", color="white", fontsize=9)
    
    # Mark best configuration
    best_idx = np.unravel_index(np.argmin(error_matrix), error_matrix.shape)
    ax.plot(best_idx[1], best_idx[0], 'r*', markersize=20, markeredgecolor='white', 
           markeredgewidth=2, label='Best')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_sweep_tradeoffs(results: Dict, save_path: str = None):
    """
    Plot trade-offs between accuracy and computational cost.
    
    Args:
        results: Output from parameter sweep
        save_path: Where to save plot
    """
    configs = results['configs']
    param_errors = results['param_errors']
    
    # Compute computational cost (proportional to num_simulations)
    costs = [c['num_simulations'] for c in configs]
    errors = [e.mean().item() for e in param_errors]
    window_sizes = [c['window_sites'] for c in configs]
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by window size
    unique_windows = sorted(list(set(window_sizes)))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_windows)))
    
    for i, w in enumerate(unique_windows):
        mask = [ws == w for ws in window_sizes]
        costs_w = [costs[j] for j in range(len(costs)) if mask[j]]
        errors_w = [errors[j] for j in range(len(errors)) if mask[j]]
        
        ax.scatter(costs_w, errors_w, c=[colors[i]], s=100, 
                  label=f'{w} sites', alpha=0.7, edgecolors='black')
    
    ax.set_xlabel('Training Simulations (computational cost)', fontsize=12)
    ax.set_ylabel('Mean Parameter Error', fontsize=12)
    ax.set_title('Accuracy vs Computational Cost', fontsize=14, fontweight='bold')
    ax.legend(title='Window Size')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Add Pareto frontier
    sorted_pairs = sorted(zip(costs, errors))
    pareto_costs = [sorted_pairs[0][0]]
    pareto_errors = [sorted_pairs[0][1]]
    
    for cost, error in sorted_pairs[1:]:
        if error < pareto_errors[-1]:  # Better than previous
            pareto_costs.append(cost)
            pareto_errors.append(error)
    
    ax.plot(pareto_costs, pareto_errors, 'r--', linewidth=2, 
           label='Pareto Frontier', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trade-off plot to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_sweep_report(results: Dict, output_dir: str = '.'):
    """
    Generate comprehensive report from parameter sweep.
    
    Args:
        results: Output from parameter sweep
        output_dir: Directory to save report and plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating sweep report in: {output_dir}")
    
    # Analyze results
    recommendations = analyze_sweep_results(results)
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_sweep_heatmap(results, os.path.join(output_dir, 'sweep_heatmap.png'))
    plot_sweep_tradeoffs(results, os.path.join(output_dir, 'sweep_tradeoffs.png'))
    
    # Save recommendations
    rec_path = os.path.join(output_dir, 'recommendations.txt')
    with open(rec_path, 'w') as f:
        f.write("PARAMETER SWEEP RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        
        f.write("OPTIMAL CONFIGURATION:\n")
        f.write(f"  num_simulations: {recommendations['optimal']['num_simulations']}\n")
        f.write(f"  window_sites: {recommendations['optimal']['window_sites']}\n\n")
        
        if recommendations['quick_test']:
            f.write("QUICK TEST CONFIGURATION:\n")
            f.write(f"  num_simulations: {recommendations['quick_test']['num_simulations']}\n")
            f.write(f"  window_sites: {recommendations['quick_test']['window_sites']}\n\n")
        
        if recommendations['production']:
            f.write("PRODUCTION CONFIGURATION:\n")
            f.write(f"  num_simulations: {recommendations['production']['num_simulations']}\n")
            f.write(f"  window_sites: {recommendations['production']['window_sites']}\n\n")
        
        f.write("USAGE EXAMPLES:\n")
        f.write(f"python snle_sliding_window_experiments.py full {recommendations['optimal']['num_simulations']}\n")
    
    print(f"\nReport saved to: {rec_path}")
    print("\n" + "="*80)
    print("SWEEP ANALYSIS COMPLETE")
    print("="*80)


# Example usage
def load_sweep_directory(sweep_dir):
    """
    Load all pickle files from a parameter sweep directory.
    
    Args:
        sweep_dir: Directory containing result_sims*.pkl files
        
    Returns:
        results: Dict with configs, param_errors, posterior_stds
    """
    import pickle
    import glob
    
    print(f"Loading sweep results from directory: {sweep_dir}")
    
    # Find all result pickle files
    result_files = glob.glob(os.path.join(sweep_dir, 'result_sims*.pkl'))
    
    if not result_files:
        raise FileNotFoundError(f"No result files found in {sweep_dir}")
    
    print(f"Found {len(result_files)} result files")
    
    configs = []
    param_errors = []
    posterior_stds = []
    
    for result_file in sorted(result_files):
        with open(result_file, 'rb') as f:
            result = pickle.load(f)
        
        # Extract configuration
        config = {
            'num_simulations': result['num_simulations'],
            'window_sites': result['window_sites']
        }
        configs.append(config)
        
        # Compute average parameter error across test cases
        test_results = result['test_results']
        
        # Collect errors from all test cases
        all_mae = []
        all_stds = []
        
        for test_result in test_results:
            if test_result is not None:
                all_mae.append(test_result['mean_mae'])
                # Use the saved posterior_std (mean across parameters)
                posterior_std = np.mean(test_result['posterior_std'])
                all_stds.append(posterior_std)
        
        if all_mae:
            # Average across test cases
            param_errors.append(torch.tensor(np.mean(all_mae)))
            posterior_stds.append(torch.tensor(np.mean(all_stds)))
        else:
            param_errors.append(torch.tensor(float('inf')))
            posterior_stds.append(torch.tensor(float('inf')))
    
    results = {
        'configs': configs,
        'param_errors': param_errors,
        'posterior_stds': posterior_stds
    }
    
    print(f"Loaded {len(configs)} configurations successfully")
    
    return results


if __name__ == "__main__":
    """
    Example workflow:
    
    1. Run parameter sweep:
       python snle_parameter_sweep.py
       
    2. Load and analyze results:
       python analyze_sweep_results.py snle_parameter_sweep/sweep_20251108_143424
    """
    
    import sys
    
    if len(sys.argv) > 1:
        # Load results from file or directory
        input_path = sys.argv[1]
        
        # Check if it's a directory with sweep results
        if os.path.isdir(input_path):
            print(f"Loading sweep directory: {input_path}")
            results = load_sweep_directory(input_path)
        else:
            # Load from single file (backwards compatibility)
            print(f"Loading from file: {input_path}")
            results = torch.load(input_path, weights_only=False)
        
        # Generate report
        output_dir = sys.argv[2] if len(sys.argv) > 2 else 'sweep_analysis'
        generate_sweep_report(results, output_dir)
        
    else:
        print("Usage: python analyze_sweep_results.py <results_path> [output_dir]")
        print("\nExamples:")
        print("  # From sweep directory:")
        print("  python analyze_sweep_results.py snle_parameter_sweep/sweep_20251108_143424")
        print("\n  # From single results file:")
        print("  python analyze_sweep_results.py sweep_results.pt sweep_analysis")
        print("\nOr run parameter sweep first:")
        print("  python snle_parameter_sweep.py")