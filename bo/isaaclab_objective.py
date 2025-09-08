#!/usr/bin/env python3
"""
Objective function for Bayesian optimization of reward functions in IsaacLab environments.

This module defines objective functions for Humanoid and Ant environments using
IsaacLab + RL-Games instead of Gymnasium + Stable-Baselines3.
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Union
import warnings
import subprocess
import os
import tempfile
import json
import time
from pathlib import Path
from .reward_function import CanonicalReward


class IsaacLabObjective:
    """
    Base objective function for optimizing IsaacLab canonical reward weights using BoTorch.
    
    This class wraps the IsaacLab + RL-Games training and evaluation pipeline to provide
    a function that can be maximized by BoTorch's optimization algorithms.
    """
    
    def __init__(self, 
                 task_name: str,  # "Isaac-Ant-v0" or "Isaac-Humanoid-v0"
                 train_timesteps: int = 300000,  # 300k timesteps as specified
                 eval_steps: int = 2000,
                 eval_num_envs: int = 16,
                 n_eval_seeds: int = 5,
                 noise_std: float = 0.1,
                 verbose: bool = False,
                 fixed_seeds: Optional[List[int]] = None,
                 isaaclab_path: str = "/home/afraz1/Documents/IsaacLab"):
        """
        Initialize the IsaacLab objective function.
        
        Args:
            task_name: IsaacLab task name ("Isaac-Ant-v0" or "Isaac-Humanoid-v0")
            train_timesteps: Total training timesteps (default: 300k)
            eval_steps: Evaluation steps per episode (default: 2000)
            eval_num_envs: Number of parallel evaluation environments (default: 16)
            n_eval_seeds: Number of random seeds to average per evaluation (default: 5)
            noise_std: Standard deviation of observation noise for GP modeling
            verbose: Whether to print detailed progress information
            fixed_seeds: Fixed seed list for reproducibility
            isaaclab_path: Path to IsaacLab installation
        """
        self.task_name = task_name
        self.train_timesteps = train_timesteps
        self.eval_steps = eval_steps
        self.eval_num_envs = eval_num_envs
        self.n_eval_seeds = n_eval_seeds
        self.noise_std = noise_std
        self.verbose = verbose
        self.isaaclab_path = Path(isaaclab_path)
        
        # Fixed seeds for reproducibility
        self.fixed_seeds = fixed_seeds if fixed_seeds is not None else [0, 1, 2, 3, 4]
        
        # Track all evaluations for analysis
        self.evaluation_history = []
        self.parameter_history = []
        
        # Determine training parameters based on task
        if "Ant" in task_name:
            self.num_envs = 128
            self.horizon_length = 16
            self.minibatch_size = 2048
        elif "Humanoid" in task_name:
            self.num_envs = 64
            self.horizon_length = 32
            self.minibatch_size = 2048
        else:
            raise ValueError(f"Unsupported task: {task_name}")
        
        # Calculate max_iterations for ~300k frames
        frames_per_iter = self.num_envs * self.horizon_length
        self.max_iterations = int(self.train_timesteps / frames_per_iter)
        
        # Get parameter bounds and dimension from canonical reward
        self.bounds = CanonicalReward.get_bounds()
        self.dim = self._get_dimension()
        
        if self.verbose:
            print(f"[ISAACLAB-OBJ] Initialized for {task_name}")
            print(f"[ISAACLAB-OBJ] Training: {self.num_envs} envs × {self.horizon_length} horizon = {frames_per_iter} frames/iter")
            print(f"[ISAACLAB-OBJ] Max iterations: {self.max_iterations} (~{self.max_iterations * frames_per_iter} frames)")
            print(f"[ISAACLAB-OBJ] Features: {self.dim} canonical features")
    
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
        Evaluate a single parameter configuration using IsaacLab + RL-Games.
        
        Args:
            weights: Array of feature weights
            
        Returns:
            Average XY distance traveled (fitness value)
        """
        # Ensure parameters are within bounds
        lower_bounds = self.bounds[0].cpu().numpy()
        upper_bounds = self.bounds[1].cpu().numpy()
        weights = np.clip(weights, lower_bounds, upper_bounds)
        
        if self.verbose:
            active_features = np.sum(np.abs(weights) > 1e-6)
            print(f"[ISAACLAB-OBJ] Evaluating weights with {active_features} active features")
            print(f"[ISAACLAB-OBJ] Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
        
        try:
            # Evaluate across multiple seeds for robustness
            distances = []
            seeds_to_use = self.fixed_seeds[:self.n_eval_seeds]
            
            for seed_idx, seed in enumerate(seeds_to_use):
                if self.verbose:
                    print(f"[ISAACLAB-OBJ] Evaluating seed {seed} ({seed_idx+1}/{len(seeds_to_use)})")
                
                distance = self._train_and_evaluate_single_seed(weights, seed)
                distances.append(distance)
                
                if self.verbose:
                    print(f"[ISAACLAB-OBJ] Seed {seed} distance: {distance:.1f}")
            
            # Average across seeds
            avg_distance = float(np.mean(distances))
            
            if self.verbose:
                print(f"[ISAACLAB-OBJ] Seeds gave distances: {[f'{d:.1f}' for d in distances]}")
                print(f"[ISAACLAB-OBJ] Average distance: {avg_distance:.3f}")
            
            # Store evaluation in history
            self.evaluation_history.append(avg_distance)
            self.parameter_history.append(weights.copy())
            
            return avg_distance
            
        except Exception as e:
            if self.verbose:
                print(f"[ISAACLAB-OBJ] Evaluation failed: {str(e)}")
            return -1000.0  # Large negative value for failed evaluations
    
    def _train_and_evaluate_single_seed(self, weights: np.ndarray, seed: int) -> float:
        """
        Train and evaluate a single seed using IsaacLab scripts.
        
        Args:
            weights: Feature weights for reward function
            seed: Random seed
            
        Returns:
            XY distance traveled
        """
        # Create temporary reward function file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            reward_code = self._generate_reward_function_code(weights)
            f.write(reward_code)
            reward_file = f.name
        
        try:
            # Create temporary directory for this run
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Step 1: Train the agent
                checkpoint_path = self._train_agent(weights, seed, temp_path, reward_file)
                
                # Step 2: Evaluate the trained agent
                distance = self._evaluate_agent(checkpoint_path, seed)
                
                return distance
                
        finally:
            # Clean up reward file
            try:
                os.unlink(reward_file)
            except:
                pass
    
    def _generate_reward_function_code(self, weights: np.ndarray) -> str:
        """Generate Python code for a custom reward function with given weights."""
        # Get feature names
        try:
            temp_reward = CanonicalReward(weights)
            feature_names = temp_reward.get_feature_names()
        except:
            feature_names = [f"feature_{i}" for i in range(len(weights))]
        
        # Generate the reward function class
        code = f'''import numpy as np
import torch
from typing import Dict, Any

class CustomReward:
    """Custom reward function generated for BO evaluation."""
    
    def __init__(self):
        # Feature weights from BO
        self.weights = np.array({weights.tolist()})
        self.feature_names = {feature_names}
    
    def __call__(self, env, prev_obs, action, obs, **kwargs):
        """Compute reward as weighted sum of features."""
        features = self.compute_features(env, prev_obs, action, obs, **kwargs)
        reward = np.dot(self.weights, features)
        return float(reward)
    
    def compute_features(self, env, prev_obs, action, obs, **kwargs):
        """Compute canonical features for {self.task_name}."""
        features = np.zeros(len(self.weights))
        
        # Extract basic features (adapt based on environment)
        if hasattr(env, 'data'):
            # MuJoCo-style environment
            qpos = env.data.qpos if hasattr(env.data, 'qpos') else obs[:len(obs)//2]
            qvel = env.data.qvel if hasattr(env.data, 'qvel') else obs[len(obs)//2:]
        else:
            # Generic observation splitting
            mid = len(obs) // 2
            qpos = obs[:mid]
            qvel = obs[mid:]
        
        # Feature 0: x_velocity (forward velocity)
        if len(features) > 0:
            features[0] = qvel[0] if len(qvel) > 0 else 0.0
        
        # Feature 1: control_cost (action magnitude penalty)
        if len(features) > 1:
            features[1] = -np.sum(np.square(action))
        
        # Feature 2: y_velocity (sideways velocity)
        if len(features) > 2:
            features[2] = qvel[1] if len(qvel) > 1 else 0.0
        
        # Feature 3: torso_height (z position)
        if len(features) > 3:
            features[3] = qpos[2] if len(qpos) > 2 else 0.0
        
        # Feature 4: angular_velocity (rotation)
        if len(features) > 4:
            features[4] = np.sum(np.abs(qvel[3:6])) if len(qvel) > 5 else 0.0
        
        # Add more features as needed...
        # For now, fill remaining features with small random values to avoid zeros
        for i in range(5, len(features)):
            if i < len(qvel):
                features[i] = qvel[i] * 0.1
            elif i < len(qpos) + len(qvel):
                features[i] = qpos[i - len(qvel)] * 0.1
            else:
                features[i] = np.random.normal(0, 0.01)
        
        return features

# Function to create the reward (for IsaacLab compatibility)
def create_reward():
    return CustomReward()
'''
        return code
    
    def _train_agent(self, weights: np.ndarray, seed: int, temp_path: Path, reward_file: str) -> str:
        """Train agent using IsaacLab RL-Games script."""
        
        # Build training command
        cmd = [
            str(self.isaaclab_path / "isaaclab.sh"), "-p",
            "scripts/reinforcement_learning/rl_games/train.py",
            "--task", self.task_name,
            "--headless",
            "--num_envs", str(self.num_envs),
            f"agent.params.config.horizon_length={self.horizon_length}",
            f"agent.params.config.minibatch_size={self.minibatch_size}",
            "--max_iterations", str(self.max_iterations),
            "--seed", str(seed)
        ]
        
        # Add reward function override if we had a way to do it
        # For now, we'll use the default reward and implement custom rewards later
        
        if self.verbose:
            print(f"[TRAIN] Running: {' '.join(cmd)}")
        
        # Run training
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
        
        result = subprocess.run(
            cmd,
            cwd=self.isaaclab_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            if self.verbose:
                print(f"[TRAIN] Training failed: {result.stderr}")
            raise RuntimeError(f"Training failed: {result.stderr}")
        
        # Find the checkpoint file
        # RL-Games saves to logs/rl_games/{task_name.lower()}/*/nn/last_*.pth
        task_log_name = self.task_name.lower().replace('-', '_')
        log_pattern = self.isaaclab_path / "logs" / "rl_games" / task_log_name
        
        # Find the most recent run directory
        run_dirs = list(log_pattern.glob("*"))
        if not run_dirs:
            raise RuntimeError("No training logs found")
        
        latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
        checkpoint_pattern = latest_run / "nn" / "last_*.pth"
        checkpoints = list(checkpoint_pattern.parent.glob("last_*.pth"))
        
        if not checkpoints:
            raise RuntimeError("No checkpoint files found")
        
        checkpoint_path = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
        
        if self.verbose:
            print(f"[TRAIN] Training completed, checkpoint: {checkpoint_path}")
        
        return checkpoint_path
    
    def _evaluate_agent(self, checkpoint_path: str, seed: int) -> float:
        """Evaluate trained agent using eval_distance.py script."""
        
        # Build evaluation command
        cmd = [
            str(self.isaaclab_path / "isaaclab.sh"), "-p",
            "scripts/evaluation/eval_distance.py",
            "--task", self.task_name,
            "--checkpoint", checkpoint_path,
            "--eval_steps", str(self.eval_steps),
            "--eval_num_envs", str(self.eval_num_envs),
            "--seed", str(seed),
            "--headless"
        ]
        
        if self.verbose:
            print(f"[EVAL] Running: {' '.join(cmd)}")
        
        # Run evaluation
        result = subprocess.run(
            cmd,
            cwd=self.isaaclab_path,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            if self.verbose:
                print(f"[EVAL] Evaluation failed: {result.stderr}")
            raise RuntimeError(f"Evaluation failed: {result.stderr}")
        
        # Parse the distance from stdout (should be a single float)
        try:
            distance = float(result.stdout.strip().split()[-1])
            return distance
        except (ValueError, IndexError) as e:
            if self.verbose:
                print(f"[EVAL] Failed to parse distance from: {result.stdout}")
            raise RuntimeError(f"Failed to parse evaluation result: {e}")
    
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
            
            if self.verbose:
                print(f"[ISAACLAB-OBJ] Evaluating point {i+1}/{batch_size}")
            
            # Evaluate this parameter configuration
            distance = self.evaluate(parameters)
            results.append(distance)
            
            if self.verbose:
                print(f"[ISAACLAB-OBJ] Point {i+1} result: {distance:.3f}")
        
        # Convert to tensor and add noise for GP modeling
        results_tensor = torch.tensor(results, dtype=torch.float64).unsqueeze(-1)
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


class AntObjective(IsaacLabObjective):
    """Objective function specifically for Isaac-Ant-v0."""
    
    def __init__(self, **kwargs):
        super().__init__(task_name="Isaac-Ant-v0", **kwargs)


class HumanoidObjective(IsaacLabObjective):
    """Objective function specifically for Isaac-Humanoid-v0."""
    
    def __init__(self, **kwargs):
        super().__init__(task_name="Isaac-Humanoid-v0", **kwargs)


def create_isaaclab_objective(task: str, **kwargs) -> IsaacLabObjective:
    """
    Factory function to create IsaacLab objective functions.
    
    Args:
        task: Task name ("ant" or "humanoid")
        **kwargs: Arguments passed to objective constructor
        
    Returns:
        Configured IsaacLab objective function
    """
    if task.lower() == "ant":
        return AntObjective(**kwargs)
    elif task.lower() == "humanoid":
        return HumanoidObjective(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task}. Use 'ant' or 'humanoid'.")


def test_isaaclab_objective(task: str = "ant", n_random_points: int = 2, verbose: bool = True) -> dict:
    """
    Test the IsaacLab objective function with random parameter values.
    
    Args:
        task: Task to test ("ant" or "humanoid")
        n_random_points: Number of random points to evaluate
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with test results
    """
    print(f"[TEST] Testing IsaacLab {task} objective function...")
    
    # Create objective function with faster settings for testing
    objective = create_isaaclab_objective(
        task=task,
        train_timesteps=50000,  # Reduced for testing (~25 iterations)
        eval_steps=1000,        # Reduced for testing
        n_eval_seeds=1,         # Single seed for testing
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
    
    if verbose:
        print(f"[TEST] Test completed!")
        print(f"[TEST] Results: {results.squeeze().tolist()}")
        if objective.evaluation_history:
            print(f"[TEST] Best distance: {max(objective.evaluation_history):.3f}")
            print(f"[TEST] Mean distance: {np.mean(objective.evaluation_history):.3f}")
    
    return {
        'test_points': X_test,
        'test_results': results,
        'objective': objective
    }

