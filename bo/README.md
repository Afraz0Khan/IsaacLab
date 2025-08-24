# Bayesian Optimization for HalfCheetah Canonical Rewards

This package implements **Bayesian Optimization (BO)** for discovering optimal reward functions in the HalfCheetah reinforcement learning environment using **canonical sum-of-features** representations.

## Key Features

✨ **Canonical Reward Functions**: Sum-of-features reward functions based on STARC framework  
🎯 **Random Initialization**: Efficient exploration with Sobol sequences in [-1, 1] bounds  
🤖 **Gaussian Process Modeling**: Advanced surrogate modeling with BoTorch  
📊 **Comprehensive Analysis**: Feature importance, convergence plots, and detailed reporting  
⚡ **Scalable Training**: PPO training with stable-baselines3 for robust evaluation  

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Canonical       │    │ Bayesian         │    │ PPO Training    │
│ Reward Function │────│ Optimization     │────│ & Evaluation    │
│ (Sum-of-Features) │    │ (BoTorch GP)     │    │ (SB3)           │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
    ┌────▼────┐             ┌────▼────┐             ┌────▼────┐
    │Features │             │Sobol    │             │Distance │
    │[-1,1]ⁿ  │             │Initial  │             │Traveled │
    │Weights  │             │Points   │             │Fitness  │
    └─────────┘             └─────────┘             └─────────┘
```

## Quick Start

### Basic Usage

```python
from bo import run_halfcheetah_bo

# Run BO with default settings
results = run_halfcheetah_bo(
    n_iterations=50,        # BO iterations after initial points
    n_initial_points=10,    # Sobol initial points  
    train_timesteps=80000,  # PPO training steps per evaluation
    eval_episodes=10,       # Evaluation episodes
    eval_timesteps=1000     # Steps per episode
)

print(f"Best distance: {results['best_value']:.4f}")
print(f"Best weights: {results['best_weights']}")
```

### Command Line Interface

```bash
# Standard optimization
python -m bo.run_optimization

# Custom parameters
python -m bo.run_optimization \
    --n-iterations 100 \
    --train-timesteps 100000 \
    --acquisition UCB \
    --exploration-factor 2.5

# Quick test (reduced parameters)
python -m bo.run_optimization --quick-test
```

## Canonical Reward Functions

The system uses **canonical sum-of-features** reward functions:

```python
reward = Σ(weight_i × feature_i)
```

### Features

Features are loaded from `starc_v2/scripts/seed_features.json` and include:

- **Velocity features**: `x_velocity`, `y_velocity`, `angular_velocities`
- **Position features**: `x_position`, `torso_height`, `joint_angles`
- **Control features**: `action_magnitude`, `action_smoothness`
- **Stability features**: `contact_forces`, `orientation_stability`

### Weight Bounds

- **All weights**: `[-1, 1]` (allows features to be turned on/off and inverted)
- **Random initialization**: Sobol sequence for efficient space coverage
- **No constraints**: BO can freely explore the weight space

### Example Canonical Reward

```python
from bo import CanonicalReward
import numpy as np

# Create reward function
reward = CanonicalReward()
print(f"Features: {len(reward.feature_names)}")

# Set random weights in [-1, 1]
weights = np.random.uniform(-1, 1, len(reward.feature_names))
reward.set_weights(weights)

# Get active features (|weight| > 1e-6)
active_features = reward.get_active_features()
print(f"Active features: {len(active_features)}")
```

## Bayesian Optimization Process

### 1. Initial Exploration (Sobol Sequence)

```python
# Generate n_initial_points using Sobol sequence
sobol_points = SobolEngine(dimension=n_features).draw(n_initial_points)
initial_weights = 2.0 * sobol_points - 1.0  # Transform to [-1, 1]
```

### 2. Evaluation Loop

For each weight vector:
1. **Create reward function** with canonical features
2. **Train PPO agent** for `train_timesteps` 
3. **Evaluate performance** over `eval_episodes`
4. **Record distance traveled** as fitness

### 3. Gaussian Process Modeling

```python
# Fit GP model to observed data
gp_model = SingleTaskGP(train_X=weights_observed, train_Y=distances_observed)
mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
fit_gpytorch_model(mll)
```

### 4. Acquisition Function Optimization

```python
# Upper Confidence Bound acquisition
acquisition = UpperConfidenceBound(model=gp_model, beta=exploration_factor)

# Optimize to find next candidate
next_weights, _ = optimize_acqf(
    acq_function=acquisition,
    bounds=torch.tensor([[-1.0] * n_features, [1.0] * n_features]),
    q=1
)
```

## Advanced Usage

### Custom Objective Function

```python
from bo import HalfCheetahObjective, BayesianOptimizer

# Create objective with custom settings
objective = HalfCheetahObjective(
    train_timesteps=100000,
    eval_episodes=15,
    eval_timesteps=1000,
    verbose=True
)

# Initialize optimizer
optimizer = BayesianOptimizer(
    objective=objective,
    n_initial_points=15,
    acquisition_function='EI',  # Expected Improvement
    exploration_factor=1.0,
    device='cuda'
)

# Run optimization
results = optimizer.run_optimization(n_iterations=75)
```

### Custom Feature Analysis

```python
# Analyze feature importance
feature_names = list(objective.get_feature_names())
active_features = objective.get_active_features(best_weights)

# Sort by importance
importance = [(name, weight) for name, weight in active_features.items()]
importance.sort(key=lambda x: abs(x[1]), reverse=True)

print("Top 5 most important features:")
for name, weight in importance[:5]:
    print(f"  {name}: {weight:+.4f}")
```

## Configuration Options

### Optimization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_iterations` | 50 | BO iterations after initial points |
| `n_initial_points` | 10 | Sobol initial points for exploration |
| `acquisition_function` | 'UCB' | Acquisition function ('UCB' or 'EI') |
| `exploration_factor` | 2.0 | Exploration parameter (β for UCB) |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_timesteps` | 80000 | PPO training steps per evaluation |
| `eval_episodes` | 10 | Episodes for fitness evaluation |
| `eval_timesteps` | 2000 | Maximum steps per evaluation episode |

### System Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | 'auto' | PyTorch device ('cpu', 'cuda', 'auto') |
| `results_dir` | 'bo_results' | Directory for saving results |
| `verbose` | True | Enable detailed progress output |

## Results and Analysis

### Optimization Results

The optimization returns a comprehensive results dictionary:

```python
{
    'best_value': 2847.3,                    # Best distance achieved
    'best_weights': [0.87, -0.23, 0.56, ...], # Optimal weight vector
    'best_features': {                       # Active features
        'x_velocity': 0.87,
        'torso_height': -0.23,
        ...
    },
    'optimization_time': 3245.7,            # Total optimization time (s)
    'total_evaluations': 60,                # Number of evaluations
    'convergence_data': {...}               # Data for plotting convergence
}
```

### Visualizations

The system automatically generates:

- **Convergence plots**: BO progress over iterations
- **Feature importance plots**: Weight magnitudes and signs  
- **Evaluation history**: Timeline of all evaluations

### Files Saved

```
bo_results/
├── bo_results_20241210_143052.json     # Main results
├── convergence_20241210_143052.png     # Convergence plot
└── feature_importance_20241210_143052.png # Feature analysis
```

## Performance Expectations

### Baseline Performance
- **Sparse reward**: ~500 distance units
- **Dense baseline**: ~1500-2000 distance units

### Optimization Targets
- **Good performance**: >2500 distance units
- **Excellent performance**: >3000 distance units
- **State-of-the-art**: >3500 distance units

### Computational Requirements

| Setting | Evaluations | Time per Eval | Total Time |
|---------|-------------|---------------|------------|
| Quick test | 8 | ~30s | ~4 minutes |
| Standard | 60 | ~90s | ~90 minutes |
| Thorough | 150 | ~90s | ~225 minutes |

*Times are approximate and depend on hardware (GPU recommended)*

## Dependencies

Core dependencies:
- `torch>=1.12.0` - PyTorch for BO and neural networks
- `botorch>=0.8.0` - Bayesian optimization framework  
- `gymnasium>=0.26.0` - RL environment interface
- `stable-baselines3>=2.0.0` - PPO implementation
- `numpy>=1.21.0` - Numerical computing
- `matplotlib>=3.5.0` - Plotting and visualization

Environment dependencies:
- `mujoco>=2.3.0` - Physics simulation for HalfCheetah
- Python 3.8+ recommended

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```python
# Use CPU device
results = run_halfcheetah_bo(device='cpu')

# Or reduce batch sizes in training
```

**Slow Convergence**  
```python
# Increase exploration
results = run_halfcheetah_bo(exploration_factor=3.0)

# Or use Expected Improvement
results = run_halfcheetah_bo(acquisition_function='EI')
```

**Poor Performance**
```python
# Increase training time
results = run_halfcheetah_bo(train_timesteps=120000)

# Or more evaluation episodes
results = run_halfcheetah_bo(eval_episodes=15)
```

### Debug Mode

```python
# Enable detailed logging
results = run_halfcheetah_bo(verbose=True)

# Test with minimal parameters
results = run_halfcheetah_bo(
    n_iterations=3,
    n_initial_points=2,
    train_timesteps=5000,
    eval_episodes=2
)
```

## Examples

See `example.py` for comprehensive demonstrations:

```bash
python -m bo.example                    # Full interactive demo
python -m bo.example simple            # Simple usage example  
python -m bo.example canonical         # Canonical reward demo
python -m bo.example objective         # Objective function demo
python -m bo.example bo                # Quick BO demo
```

## Contributing

The framework is modular and extensible:

- **New acquisition functions**: Extend `BayesianOptimizer.get_acquisition_function()`
- **New features**: Add to `seed_features.json` and update `CanonicalReward`
- **New environments**: Implement new objective functions following `HalfCheetahObjective`
- **Analysis tools**: Extend utilities in `utils.py`

---

🚀 **Ready to optimize?** Start with `python -m bo.run_optimization --quick-test` 