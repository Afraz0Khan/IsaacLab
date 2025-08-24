#!/usr/bin/env python3
"""
Objective function for Bayesian optimization of reward functions.

This module defines the objective function that BoTorch will optimize,
wrapping the training and evaluation pipeline for canonical reward functions.
"""

import torch
import numpy as np
from typing import Optional, Dict, List
import warnings
from .train_evaluate import train_and_evaluate
from .reward_function import CanonicalReward


def _parallel_evaluation_worker(args):
    """
    Worker function for parallel evaluation of different seeds.
    This needs to be at module level to be picklable by multiprocessing.
    """
    seed_hash, parameters, train_timesteps, eval_episodes, eval_timesteps, use_gpu, n_envs, fitness_metric, expert_weights = args
    return train_and_evaluate(
        parameters=parameters,
        train_timesteps=train_timesteps,
        eval_episodes=eval_episodes,
        eval_timesteps=eval_timesteps,
        seed=seed_hash,
        verbose=True,  # Enable progress bars for parallel runs
        use_gpu=use_gpu,
        n_envs=n_envs,
        fitness_metric=fitness_metric,
        expert_weights=expert_weights
    )


class HalfCheetahObjective:
    """
    Objective function for optimizing HalfCheetah canonical reward weights using BoTorch.
    
    This class wraps the training and evaluation pipeline to provide a function
    that can be maximized by BoTorch's optimization algorithms.
    """
    
    def __init__(self, 
                 train_timesteps: int = 80000,
                 eval_episodes: int = 10,
                 eval_timesteps: int = 2000,
                 noise_std: float = 0.1,
                 n_eval_seeds: int = 5,
                 verbose: bool = False,
                 use_gpu: bool = False,
                 fixed_seeds: Optional[List[int]] = None,
                 n_envs: int = 1,
                 fitness_metric: str = "distance",
                 expert_weights: Optional[np.ndarray] = None):
        """
        Initialize the objective function.
        
        Args:
            train_timesteps: Number of timesteps for training each agent
            eval_episodes: Number of episodes for evaluation
            eval_timesteps: Maximum steps per evaluation episode
            noise_std: Standard deviation of observation noise (for GP modeling)
            n_eval_seeds: Number of random seeds to average per evaluation
            verbose: Whether to print detailed progress information
            use_gpu: Whether to use GPU for training (default: False, uses CPU)
        """
        self.train_timesteps = train_timesteps
        self.eval_episodes = eval_episodes
        self.eval_timesteps = eval_timesteps
        self.noise_std = noise_std
        self.n_eval_seeds = n_eval_seeds
        self.verbose = verbose
        self.use_gpu = use_gpu
        # Hardcoded seed schedule for consistent experiments
        self.fixed_seeds = fixed_seeds if fixed_seeds is not None else [0, 1, 2, 3, 4]
        
        # Track all evaluations for analysis
        self.evaluation_history = []
        self.parameter_history = []
        # Vectorized envs per PPO to saturate CPU
        self.n_envs = max(1, int(n_envs))
        # Fitness configuration
        self.fitness_metric = fitness_metric
        self.expert_weights = expert_weights
        
        # Get parameter bounds and dimension from canonical reward
        self.bounds = CanonicalReward.get_bounds()
        self.dim = self._get_dimension()
        
        if self.verbose:
            print(f"[OBJECTIVE] Initialized with {self.dim} canonical features")
            
    def _get_dimension(self) -> int:
        """Get dimension from canonical reward function."""
        try:
            # Create a temporary canonical reward to get dimension
            temp_weights = np.zeros(21)  # Try with default size first
            temp_reward = CanonicalReward(temp_weights)
            return temp_reward.dim
        except Exception:
            # Fallback to bounds dimension
            return self.bounds.shape[1]
    
    def get_feature_count(self) -> int:
        """Get the number of features in the canonical reward function."""
        return self.dim
    
    def get_feature_names(self) -> list:
        """Return the list of canonical feature names."""
        try:
            temp_weights = np.zeros(self.dim)
            temp_reward = CanonicalReward(temp_weights)
            return temp_reward.get_feature_names()
        except Exception:
            # Fallback: generic feature names as list
            return [f"feature_{i}" for i in range(self.dim)]
    
    def get_active_features(self, weights: np.ndarray) -> Dict[str, float]:
        """Get active features (non-zero weights) with their values."""
        feature_names = self.get_feature_names()
        active = {}
        for i, name in enumerate(feature_names):
            if i < len(weights) and abs(weights[i]) > 1e-6:
                active[name] = float(weights[i])
        return active
    
    def evaluate(self, weights: np.ndarray) -> float:
        """
        Evaluate a single parameter configuration.
        
        Args:
            weights: Array of feature weights
            
        Returns:
            Distance traveled (fitness value)
        """
        # Ensure parameters are within bounds (clip if necessary)
        lower_bounds = self.bounds[0].cpu().numpy()
        upper_bounds = self.bounds[1].cpu().numpy()
        weights = np.clip(weights, lower_bounds, upper_bounds)
        
        if self.verbose:
            active_features = np.sum(np.abs(weights) > 1e-6)
            print(f"[OBJECTIVE] Evaluating weights with {active_features} active features")
            print(f"[OBJECTIVE] Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
        
        # Evaluate this parameter configuration using multi-seed evaluation
        try:
            n_seeds = self.n_eval_seeds
            
            if n_seeds == 1:
                # Single seed - no parallelization needed
                import hashlib
                param_str = str(weights.round(6)) + "_seed_0"
                seed_hash = int(hashlib.md5(param_str.encode()).hexdigest()[:8], 16) % 10000
                
                distance = train_and_evaluate(
                    parameters=weights,
                    train_timesteps=self.train_timesteps,
                    eval_episodes=self.eval_episodes,
                    eval_timesteps=self.eval_timesteps,
                    seed=seed_hash,
                    verbose=self.verbose,
                    use_gpu=self.use_gpu,
                    n_envs=self.n_envs,
                    fitness_metric=self.fitness_metric,
                    expert_weights=self.expert_weights
                )
            else:
                # Multiple seeds - use parallel evaluation
                distances = self._evaluate_parallel_seeds(weights, n_seeds)
                distance = float(np.mean(distances))
                
                # Always print individual seed distances for transparency
                print(f"[OBJECTIVE] Seeds gave distances: {[f'{d:.1f}' for d in distances]}")
                print(f"[OBJECTIVE] Average distance: {distance:.3f}")
            
            # Store evaluation in history
            self.evaluation_history.append(distance)
            self.parameter_history.append(weights.copy())
            
        except Exception as e:
            if self.verbose:
                print(f"[OBJECTIVE] Evaluation failed: {str(e)}")
            distance = -1000.0  # Large negative value for failed evaluations
        
        return distance

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the objective function at given parameter values.
        
        Args:
            X: Tensor of shape (n_points, dim) containing parameter values to evaluate
            
        Returns:
            Tensor of shape (n_points, 1) containing objective values (distances)
        """
        # Handle single point evaluation
        if X.dim() == 1:
            X = X.unsqueeze(0)
        
        batch_size = X.shape[0]
        results = []
        
        for i in range(batch_size):
            parameters = X[i].detach().cpu().numpy()
            
            # Ensure parameters are within bounds (clip if necessary)
            lower_bounds = self.bounds[0].cpu().numpy()
            upper_bounds = self.bounds[1].cpu().numpy()
            parameters = np.clip(parameters, lower_bounds, upper_bounds)
            
            if self.verbose:
                print(f"[OBJECTIVE] Evaluating point {i+1}/{batch_size}")
                active_features = np.sum(np.abs(parameters) > 1e-6)
                print(f"[OBJECTIVE] Active features: {active_features}/{len(parameters)}")
                print(f"[OBJECTIVE] Weight range: [{parameters.min():.3f}, {parameters.max():.3f}]")
            
            # Evaluate this parameter configuration
            try:
                # Evaluate with multiple seeds and average for robustness
                n_seeds = self.n_eval_seeds
                
                if n_seeds == 1:
                    # Single seed - no parallelization needed
                    import hashlib
                    param_str = str(parameters.round(6)) + "_seed_0"
                    seed_hash = int(hashlib.md5(param_str.encode()).hexdigest()[:8], 16) % 10000
                    
                    distance = train_and_evaluate(
                        parameters=parameters,
                        train_timesteps=self.train_timesteps,
                        eval_episodes=self.eval_episodes,
                        eval_timesteps=self.eval_timesteps,
                        seed=seed_hash,
                        verbose=self.verbose,
                        use_gpu=self.use_gpu,
                        n_envs=self.n_envs,
                        fitness_metric=self.fitness_metric,
                        expert_weights=self.expert_weights
                    )
                else:
                    # Multiple seeds - use parallel evaluation
                    distances = self._evaluate_parallel_seeds(parameters, n_seeds)
                    distance = float(np.mean(distances))
                    
                    # Always print individual seed distances for transparency
                    print(f"[OBJECTIVE] Seeds gave distances: {[f'{d:.1f}' for d in distances]}")
                    print(f"[OBJECTIVE] Average distance: {distance:.3f}")
                
                # Store evaluation in history
                self.evaluation_history.append(distance)
                self.parameter_history.append(parameters.copy())
                
            except Exception as e:
                if self.verbose:
                    print(f"[OBJECTIVE] Evaluation failed: {str(e)}")
                distance = -1000.0  # Large negative value for failed evaluations
            
            results.append(distance)
            
            if self.verbose:
                print(f"[OBJECTIVE] Point {i+1} result: {distance:.3f}")
        
        # Convert to tensor and add noise for GP modeling
        results_tensor = torch.tensor(results, dtype=torch.float64).unsqueeze(-1)
        # Add small amount of noise to avoid numerical issues with GP
        if self.noise_std > 0:
            noise = torch.randn_like(results_tensor) * self.noise_std
            results_tensor = results_tensor + noise
            
        return results_tensor
    
    def get_bounds(self) -> torch.Tensor:
        """Get parameter bounds for optimization."""
        return self.bounds
    
    def get_best_parameters(self) -> Optional[np.ndarray]:
        """Get the best parameters found so far."""
        if not self.evaluation_history:
            return None
        
        best_idx = np.argmax(self.evaluation_history)
        return self.parameter_history[best_idx].copy()
    
    def get_best_value(self) -> Optional[float]:
        """Get the best objective value found so far."""
        if not self.evaluation_history:
            return None
        
        return float(np.max(self.evaluation_history))
    
    def get_evaluation_history(self) -> list:
        """Get history of all objective evaluations."""
        return self.evaluation_history.copy()
    
    def reset_history(self):
        """Clear evaluation history."""
        self.evaluation_history = []
        self.parameter_history = []
    
    def summary_stats(self) -> dict:
        """Get summary statistics of evaluations."""
        if not self.evaluation_history:
            return {}
        
        evaluations = np.array(self.evaluation_history)
        return {
            'n_evaluations': len(evaluations),
            'best_value': float(np.max(evaluations)),
            'mean_value': float(np.mean(evaluations)),
            'std_value': float(np.std(evaluations)),
            'min_value': float(np.min(evaluations)),
        }
    
    def get_feature_analysis(self) -> dict:
        """Analyze which features are most important based on evaluations."""
        if not self.evaluation_history or not self.parameter_history:
            return {}
        
        try:
            # Get feature names from canonical reward
            temp_reward = CanonicalReward(np.zeros(self.dim))
            feature_names = temp_reward.get_feature_names()
        except:
            feature_names = [f"feature_{i}" for i in range(self.dim)]
        
        # Convert to arrays for analysis
        parameters_array = np.array(self.parameter_history)
        evaluations_array = np.array(self.evaluation_history)
        
        # Calculate correlation between each feature weight and objective value
        feature_correlations = {}
        for i, feature_name in enumerate(feature_names):
            if parameters_array.shape[1] > i:
                correlation = np.corrcoef(parameters_array[:, i], evaluations_array)[0, 1]
                if not np.isnan(correlation):
                    feature_correlations[feature_name] = correlation
        
        return {
            'feature_correlations': feature_correlations,
            'feature_names': feature_names,
            'most_positive_features': sorted(feature_correlations.items(), 
                                           key=lambda x: x[1], reverse=True)[:5],
            'most_negative_features': sorted(feature_correlations.items(), 
                                           key=lambda x: x[1])[:5]
        }
    
    def _evaluate_parallel_seeds(self, parameters: np.ndarray, n_seeds: int) -> List[float]:
        """Evaluate multiple seeds in parallel using optimized multiprocessing."""
        import multiprocessing as mp
        import functools
        
        # Use fixed seeds for consistency across experiments
        seeds = self._get_eval_seeds(n_seeds)
        
        if self.verbose:
            print(f"[OBJECTIVE] Parallel evaluation with {n_seeds} seeds: {seeds}")
            print(f"[OBJECTIVE] Using {n_seeds} CPU processes")
        
        # Prepare arguments for the module-level worker function
        worker_args = [
            (seed, parameters, self.train_timesteps, self.eval_episodes, self.eval_timesteps, self.use_gpu, self.n_envs, self.fitness_metric, self.expert_weights)
            for seed in seeds
        ]
        
        # Use spawn to avoid fork-with-MuJoCo issues
        try:
            ctx = mp.get_context('spawn')
        except Exception:
            ctx = mp
            
        # Use a persistent pool if we have multiple evaluations to reduce overhead
        if not hasattr(self, '_process_pool') or self._process_pool is None:
            self._process_pool = ctx.Pool(processes=min(n_seeds, mp.cpu_count()))
            
        distances = self._process_pool.map(_parallel_evaluation_worker, worker_args)
        
        # Always show individual seed results for transparency
        print(f"[OBJECTIVE] Individual seed distances: {[f'{d:.1f}' for d in distances]}")
        
        return distances

    def _get_eval_seeds(self, n: int) -> List[int]:
        """Return the first n seeds from the fixed seed list, cycling if needed."""
        if n <= len(self.fixed_seeds):
            return self.fixed_seeds[:n]
        # Cycle deterministically if more seeds requested
        out = []
        idx = 0
        while len(out) < n:
            out.append(self.fixed_seeds[idx % len(self.fixed_seeds)])
            idx += 1
        return out

    def evaluate_with_seeds(self, parameters: np.ndarray, seeds: List[int]) -> List[float]:
        """Evaluate a parameter configuration over an explicit seed list (parallel)."""
        return self._evaluate_parallel_seeds(parameters, n_seeds=len(seeds))
        
    def __del__(self):
        """Clean up the process pool when the objective is destroyed."""
        if hasattr(self, '_process_pool') and self._process_pool is not None:
            self._process_pool.close()
            self._process_pool.join()


class NoisyHalfCheetahObjective(HalfCheetahObjective):
    """
    Noisy version of the objective function that adds explicit noise to evaluations.
    
    This can be useful for testing BO algorithms' robustness to noise or when
    you want to model evaluation uncertainty explicitly.
    """
    
    def __init__(self, evaluation_noise_std: float = 50.0, **kwargs):
        """
        Initialize noisy objective function.
        
        Args:
            evaluation_noise_std: Standard deviation of evaluation noise
            **kwargs: Arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.evaluation_noise_std = evaluation_noise_std
    
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate with added noise."""
        # Get base evaluation
        base_results = super().__call__(X)
        
        # Add evaluation noise
        if self.evaluation_noise_std > 0:
            eval_noise = torch.randn_like(base_results) * self.evaluation_noise_std
            return base_results + eval_noise
        
        return base_results


def create_objective(objective_type: str = "standard", **kwargs) -> HalfCheetahObjective:
    """
    Factory function to create objective functions.
    
    Args:
        objective_type: Type of objective ("standard" or "noisy")
        **kwargs: Arguments passed to objective constructor
        
    Returns:
        Configured objective function
    """
    if objective_type == "standard":
        return HalfCheetahObjective(**kwargs)
    elif objective_type == "noisy":
        return NoisyHalfCheetahObjective(**kwargs)
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")


def test_objective_function(n_random_points: int = 3, verbose: bool = True) -> dict:
    """
    Test the objective function with random parameter values.
    
    Args:
        n_random_points: Number of random points to evaluate
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with test results
    """
    print("[TEST] Testing canonical objective function with random parameters...")
    
    # Create objective function with faster settings for testing
    objective = HalfCheetahObjective(
        train_timesteps=20000,  # Reduced for testing
        eval_episodes=5,        # Reduced for testing
        eval_timesteps=2000,
        verbose=verbose
    )
    
    # Get bounds and generate random points
    bounds = objective.get_bounds()
    
    # Generate random points within bounds
    random_points = []
    for i in range(n_random_points):
        point = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(objective.dim)
        random_points.append(point)
    
    X_test = torch.stack(random_points)
    
    if verbose:
        print(f"[TEST] Evaluating {n_random_points} random parameter configurations...")
        print(f"[TEST] Parameter dimension: {objective.dim}")
        print(f"[TEST] Weight bounds: [{bounds[0][0]:.1f}, {bounds[1][0]:.1f}]")
    
    # Evaluate the random points
    results = objective(X_test)
    
    # Get summary
    stats = objective.summary_stats()
    
    # Get feature analysis
    feature_analysis = objective.get_feature_analysis()
    
    if verbose:
        print(f"[TEST] Test completed!")
        print(f"[TEST] Results: {results.squeeze().tolist()}")
        print(f"[TEST] Best distance: {stats.get('best_value', 'N/A'):.3f}")
        print(f"[TEST] Mean distance: {stats.get('mean_value', 'N/A'):.3f}")
        
        if feature_analysis and feature_analysis.get('most_positive_features'):
            print("[TEST] Most positively correlated features:")
            for name, corr in feature_analysis['most_positive_features']:
                print(f"[TEST]   {name}: {corr:.3f}")
    
    return {
        'test_points': X_test,
        'test_results': results,
        'statistics': stats,
        'feature_analysis': feature_analysis,
        'objective': objective
    } 