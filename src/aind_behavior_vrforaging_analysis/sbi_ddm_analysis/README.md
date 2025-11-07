# Clean Patch Foraging SBI Inference

Simplified, readable implementation of SBI inference for patch foraging DDM.

## File Structure (4 core files, ~400 lines total)

```
simulator.py    # Simulate behavior (150 lines)
features.py           # Extract features from windows (100 lines)  
inference.py          # Train SBI and infer parameters (100 lines)
validation.py         # Diagnostics and metrics (150 lines)
experiments.py        # Run experiments (100 lines)
```

## Quick Start

### Train and Test

```python
from simulator import PatchForagingDDM, create_prior
from inference import train_sbi, infer_parameters

# Train
simulator = PatchForagingDDM()
prior = create_prior()
posterior = train_sbi(simulator, prior, num_simulations=50000)

# Infer on new data
test_window = simulator.simulate_with_random_walk(test_theta, window_sites=100)
samples = infer_parameters(posterior, test_window, num_samples=1000)
```

### Run Experiments

```bash
# Quick test (5K simulations)
python experiments.py quick

# Full pipeline (50K simulations + validation)
python experiments.py full

# Default (50K simulations)
python experiments.py
```

## Data Flow

```
1. Simulator generates behavior
   window = simulator.simulate_with_random_walk(theta, 100)  # → (100, 3)

2. Extract features
   features = extract_features(window)  # → (611,)

3. Train posterior
   posterior = train_sbi(simulator, prior, 50000)

4. Infer parameters
   samples = infer_parameters(posterior, window)  # → (1000, 3)
```

## Features Extracted (611 total)

```
Original data (flattened):      300
Inter-site intervals:           100  ← Key for drift_rate
Cumulative rewards per patch:   100  ← Tracks reward probability
Failure run lengths:            100  ← Isolates failure bump
Summary statistics:              11  ← Robust aggregates
```

## Running Validation

```python
from validation import run_sbc, print_correlations, plot_pairplot

# Check calibration
ranks = run_sbc(simulator, prior, posterior, num_tests=50)

# Check parameter separation
test_window = simulator.simulate_with_random_walk(theta, 100)
samples = infer_parameters(posterior, test_window)
print_correlations(samples)  # Want r < 0.5 for drift ↔ reward
plot_pairplot(samples, true_theta=theta, save_path='pairplot.png')
```

## Testing

Each module is self-contained and testable:

```bash
python features.py        # Test feature extraction
python inference.py       # Test SBI training
python validation.py      # Test diagnostics
python simulator.py       # Test simulator
```

## Common Tasks

### Generate training data
```python
from inference import generate_training_data
thetas, features = generate_training_data(simulator, prior, 10000)
```

### Check if parameters are identifiable
```python
from validation import print_correlations
# After getting posterior samples:
print_correlations(samples)  # Shows correlation matrix
```

### Validate recovery
```python
from validation import run_sbc
ranks = run_sbc(simulator, prior, posterior, num_tests=50)
# Ranks should be ~uniform if well-calibrated
```

## Key Result to Check

After training, check the correlation between `drift_rate` and `reward_bump`:

```python
samples = infer_parameters(posterior, test_window, num_samples=2000)
corr_matrix = np.corrcoef(samples.numpy().T)
print(f"drift ↔ reward correlation: {corr_matrix[0, 1]:.3f}")
```