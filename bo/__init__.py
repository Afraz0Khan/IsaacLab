"""
Bayesian Optimization for HalfCheetah Canonical Reward Functions.

This package implements Bayesian optimization for discovering optimal reward functions
in the HalfCheetah reinforcement learning environment using canonical sum-of-features
representations with weights bounded in [-1, 1].

Main Components:
- CanonicalReward: Sum-of-features reward function implementation
- HalfCheetahObjective: Objective function for BO
- BayesianOptimizer: BO implementation with Gaussian Process modeling
- Load existing STARC seed rewards as initial population
- Utility functions for analysis and visualization

Example Usage:
    from bo import run_halfcheetah_bo
    
    # Use STARC seed rewards as initial population
    results = run_halfcheetah_bo(
        n_iterations=50,
        use_seed_rewards=True,
        train_timesteps=80000
    )
    
    # Or use random initialization  
    results = run_halfcheetah_bo(
        n_iterations=50,
        n_initial_points=10,
        train_timesteps=80000
    )
"""

from .reward_function import CanonicalReward
from .objective import HalfCheetahObjective
from .bayesian_optimization import BayesianOptimizer  
from .run_optimization import run_halfcheetah_bo
from .train_evaluate import train_and_evaluate, test_baseline_performance
from .load_existing_rewards import (
    load_seed_rewards_as_initial_points,
    plot_seed_rewards,
    get_seed_reward_info
)

__version__ = "1.0.0"
__author__ = "BO Framework"

__all__ = [
    # Main functions
    'run_halfcheetah_bo',
    
    # Core classes
    'CanonicalReward',
    'HalfCheetahObjective', 
    'BayesianOptimizer',
    
    # Training functions
    'train_and_evaluate',
    'test_baseline_performance',
    
    # Seed reward functions
    'load_seed_rewards_as_initial_points',
    'plot_seed_rewards',
    'get_seed_reward_info',
] 