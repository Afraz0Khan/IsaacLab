#!/usr/bin/env python3
"""
Training and evaluation pipeline for Bayesian optimization.

This module handles training PPO agents with custom reward functions and 
evaluating their performance on the HalfCheetah environment.
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from typing import List, Tuple, Optional
import tempfile
import shutil
import os
import time

from .reward_function import CanonicalReward


class RewardWrapper(gym.Wrapper):
    """Wrapper that replaces the environment's reward with a custom reward function."""
    
    def __init__(self, env: gym.Env, reward_fn: CanonicalReward):
        super().__init__(env)
        self.reward_fn = reward_fn
        self._last_obs = None
        self._last_x_pos = 0.0
        self.episode_distances = []
        self.start_x = 0.0
        
    def reset(self, **kwargs):
        reset_res = self.env.reset(**kwargs)
        if isinstance(reset_res, tuple):
            obs, info = reset_res
        else:
            obs, info = reset_res, {}
        self._last_obs = obs
        
        # Get starting x position (prefer Mujoco state; fallback to info)
        self.start_x = self._get_x_position(info)
        self._last_x_pos = self.start_x
        
        return obs, info
    
    def step(self, action):
        state = self._last_obs
        step_res = self.env.step(action)
        if isinstance(step_res, tuple) and len(step_res) == 5:
            obs, original_reward, terminated, truncated, info = step_res
        else:
            # Legacy 4-tuple: (obs, reward, done, info)
            obs, original_reward, done, info = step_res
            terminated = bool(done)
            truncated = False
        
        # Get current x position (prefer Mujoco state)
        current_x = self._get_x_position(info)
        
        # Calculate x_velocity from position change if not provided
        x_velocity = info.get("x_velocity", None)
        if x_velocity is None:
            dt = self._get_dt()
            x_velocity = (current_x - self._last_x_pos) / dt if dt > 0 else 0.0
        
        # Calculate custom reward
        custom_reward = self.reward_fn(
            env=self.env,
            state=state, 
            action=action,
            next_state=obs,
            x_velocity=x_velocity
        )
        
        # Store episode distance when episode ends
        if terminated or truncated:
            episode_distance = current_x - self.start_x
            self.episode_distances.append(episode_distance)
        
        # Update for next step
        self._last_obs = obs
        self._last_x_pos = current_x
        
        return obs, float(custom_reward), terminated, truncated, info

    def _get_x_position(self, info: dict) -> float:
        # Try Mujoco qpos[0] from unwrapped env
        try:
            env_unwrapped = self.env.unwrapped
            if hasattr(env_unwrapped, "data") and hasattr(env_unwrapped.data, "qpos"):
                return float(env_unwrapped.data.qpos[0])
        except Exception:
            pass
        return float(info.get("x_position", 0.0))

    def _get_dt(self) -> float:
        try:
            env_unwrapped = self.env.unwrapped
            if hasattr(env_unwrapped, "dt"):
                return float(env_unwrapped.dt)
        except Exception:
            pass
        return float(getattr(self.env, "dt", 0.05))
    
    def get_episode_distances(self) -> List[float]:
        """Get distances traveled in completed episodes."""
        return self.episode_distances.copy()
    
    def clear_distances(self):
        """Clear stored episode distances."""
        self.episode_distances = []


def make_halfcheetah_env(max_episode_steps: int = 1000, render_mode: Optional[str] = None) -> gym.Env:
    """Create HalfCheetah environment with TimeLimit wrapper."""
    try:
        # Try to use custom HalfCheetah implementation first
        from custom_gym_envs.envs.half_cheetah.HalfCheetah_V1.half_cheetah import HalfCheetahEnv
        env = HalfCheetahEnv(render_mode=render_mode) if render_mode is not None else HalfCheetahEnv()
    except ImportError:
        # Fallback to standard Gymnasium HalfCheetah
        if render_mode is None:
            env = gym.make("HalfCheetah-v4")
        else:
            env = gym.make("HalfCheetah-v4", render_mode=render_mode)
    
    # Apply finite bounds wrapper for stability
    if isinstance(env.observation_space, gym.spaces.Box):
        low = np.where(np.isinf(env.observation_space.low), -1e10, env.observation_space.low)
        high = np.where(np.isinf(env.observation_space.high), 1e10, env.observation_space.high)
        env.observation_space = gym.spaces.Box(low.astype(np.float32), high.astype(np.float32), dtype=np.float32)
    
    # Apply episode length limit
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    
    return env


def train_and_evaluate(parameters: np.ndarray, 
                      train_timesteps: int = 80000,
                      eval_episodes: int = 10, 
                      eval_timesteps: int = 2000,
                      seed: Optional[int] = None,
                      verbose: bool = False,
                      use_gpu: bool = False,
                      n_envs: int = 1,
                      fitness_metric: str = "distance",
                      expert_weights: Optional[np.ndarray] = None) -> float:
    """
    Train a PPO agent with given reward parameters and evaluate its performance.
    
    Training process:
    - Trains for train_timesteps total timesteps
    - Policy is evaluated every 2000 timesteps during training (monitoring)
    - Final evaluation uses eval_episodes episodes for fitness score
    
    Args:
        parameters: Array of reward function parameters (weights for canonical features)
        train_timesteps: Number of training timesteps (default: 80000)
        eval_episodes: Number of evaluation episodes for final fitness (default: 10)
        eval_timesteps: Maximum steps per evaluation episode (default: 5000)
        seed: Random seed for reproducibility
        verbose: Whether to print progress information
        use_gpu: Whether to use GPU for training (default: False, uses CPU)
        
    Returns:
        Fitness value according to fitness_metric ("distance" -> avg distance; "expert" -> avg expert reward)
    """
    if verbose:
        print(f"[TRAIN] Starting training with {len(parameters)} feature weights")
        print(f"[TRAIN] Weight range: [{parameters.min():.3f}, {parameters.max():.3f}]")
        start_time = time.time()
    
    # Create reward function
    try:
        reward_fn = CanonicalReward(parameters)
        if verbose:
            print(f"[TRAIN] Created canonical reward with {reward_fn.dim} features")
            print(f"[TRAIN] Active features: {sum(abs(w) > 1e-6 for w in parameters)}")
    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to create reward function: {e}")
        return -1000.0
    
    # Create training environment (optionally vectorized for higher CPU throughput)
    if n_envs and n_envs > 1:
        def _make_train_env():
            def thunk():
                env = make_halfcheetah_env(max_episode_steps=eval_timesteps)
                env = RewardWrapper(env, reward_fn)
                return Monitor(env)
            return thunk
        # Use subprocess-based vectorization to leverage multiple CPU cores
        train_env = SubprocVecEnv([_make_train_env() for _ in range(int(n_envs))], start_method="fork")
    else:
        train_env = make_halfcheetah_env(max_episode_steps=eval_timesteps)
        train_env = RewardWrapper(train_env, reward_fn)
        train_env = Monitor(train_env)
    
    # Create evaluation environment
    eval_env = make_halfcheetah_env(max_episode_steps=eval_timesteps)
    eval_reward_fn = CanonicalReward(parameters)  # Fresh instance for eval
    eval_reward_wrapper = RewardWrapper(eval_env, eval_reward_fn)  # Store reference
    eval_env = Monitor(eval_reward_wrapper)
    
    try:
        # Train PPO agent
        if verbose:
            print(f"[TRAIN] Training PPO for {train_timesteps} timesteps...")
            print(f"[TRAIN] Policy will be evaluated every 2000 timesteps ({train_timesteps//2000} evaluations during training)")
            print(f"[TRAIN] Final evaluation uses {eval_episodes} episodes for BO fitness score")
            
        # Use specified device (GPU or CPU)
        # Default is CPU (use_gpu=False) to avoid the GPU warning for MLP policies
        import torch
        if use_gpu and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0,  # Always quiet to avoid training tables
            seed=seed,
            n_steps=2048,        # Keep original for consistency
            batch_size=64,       # Keep original for consistency  
            learning_rate=3e-4,  # Keep original for consistency
            device=device,       # Only change device, not hyperparameters
            tensorboard_log=None  # Disable tensorboard for BO runs
        )
        
        # Set up evaluation callback for monitoring during training
        from stable_baselines3.common.callbacks import EvalCallback
        import tempfile
        import os
        
        # Create temporary directory for eval callback logs
        temp_dir = tempfile.mkdtemp(prefix="ppo_eval_")
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=None,  # Don't save models during BO
            log_path=temp_dir,
            eval_freq=2000,  # Keep original: evaluate every 2000 timesteps
            n_eval_episodes=3,  # Keep original: 3 episodes per evaluation
            deterministic=True,
            verbose=0,  # Keep quiet during intermediate evaluations
            warn=False
        )
        
        # Train the model with evaluation callback
        model.learn(
            total_timesteps=train_timesteps, 
            progress_bar=verbose,
            callback=eval_callback
        )
        
        # Clean up temporary directory
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        
        if verbose:
            train_time = time.time() - start_time
            print(f"[TRAIN] Training completed in {train_time:.1f}s")
        
        # Final evaluation phase (for BO fitness score)
        if verbose:
            print(f"[EVAL] Final evaluation over {eval_episodes} episodes...")
            eval_start = time.time()
        
        # Clear any distances from intermediate evaluations
        eval_reward_wrapper.clear_distances()
        
        episode_rewards, episode_lengths = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=eval_episodes,
            deterministic=True,
            return_episode_rewards=True
        )
        
        # Get episode distances from the reward wrapper directly
        episode_distances = eval_reward_wrapper.get_episode_distances()
        
        # Calculate average distance traveled
        if len(episode_distances) > 0:
            avg_distance = np.mean(episode_distances)
        else:
            # Fallback: estimate distance from episode info if available
            avg_distance = 0.0
            if hasattr(eval_env, 'get_wrapper_attr'):
                try:
                    episode_infos = eval_env.get_wrapper_attr('episode_infos')
                    if episode_infos:
                        distances = [info.get('episode_distance', 0.0) for info in episode_infos]
                        if distances:
                            avg_distance = np.mean(distances)
                except:
                    pass
        
        if verbose:
            eval_time = time.time() - eval_start
            total_time = time.time() - start_time
            print(f"[EVAL] Evaluation completed in {eval_time:.1f}s")
            print(f"[RESULT] Average distance: {avg_distance:.3f}")
            print(f"[RESULT] Average reward: {np.mean(episode_rewards):.3f}")
            print(f"[RESULT] Average length: {np.mean(episode_lengths):.1f}")
            print(f"[RESULT] Total time: {total_time:.1f}s")
            print("-" * 50)
        
        # Decide fitness according to metric
        metric_value: float
        if fitness_metric == "distance":
            metric_value = float(avg_distance)
        elif fitness_metric == "expert":
            # Build expert reward weights if not provided
            if expert_weights is None:
                try:
                    temp_reward = CanonicalReward(np.zeros_like(parameters))
                except Exception:
                    temp_reward = CanonicalReward(parameters)
                feature_names = temp_reward.get_feature_names()
                ew = np.zeros(len(feature_names), dtype=float)
                for i, name in enumerate(feature_names):
                    if name in ["x_velocity", "forward_velocity"]:
                        ew[i] = 1.0
                    elif name in ["ctrl_cost", "control_cost"]:
                        ew[i] = -0.1
                expert_weights = ew

            # Evaluate trained policy on expert reward
            from stable_baselines3.common.monitor import Monitor as _Monitor
            expert_env = make_halfcheetah_env(max_episode_steps=eval_timesteps)
            expert_env = RewardWrapper(expert_env, CanonicalReward(expert_weights))
            expert_env = _Monitor(expert_env)
            exp_rewards, _ = evaluate_policy(
                model, expert_env, n_eval_episodes=eval_episodes, deterministic=True, return_episode_rewards=True
            )
            expert_env.close()
            metric_value = float(np.mean(exp_rewards))
        else:
            # Unknown metric; default to distance
            metric_value = float(avg_distance)

        # Clean up model to free memory
        del model

        return metric_value
    
    except Exception as e:
        if verbose:
            print(f"[ERROR] Training/evaluation failed: {str(e)}")
            print(f"[ERROR] Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
        
        # Clean up on error
        try:
            if 'model' in locals():
                del model
        except:
            pass
            
        return -1000.0  # Return large negative value for failed runs
        
    finally:
        # Clean up environments
        try:
            train_env.close()
            eval_env.close()
            # Clear any stored references
            if 'eval_reward_wrapper' in locals():
                eval_reward_wrapper.clear_distances()
            # Force garbage collection
            import gc
            gc.collect()
        except Exception as cleanup_error:
            if verbose:
                print(f"[WARNING] Environment cleanup failed: {cleanup_error}")


def batch_evaluate(parameter_list: List[np.ndarray], 
                  train_timesteps: int = 80000,
                  eval_episodes: int = 10,
                                               eval_timesteps: int = 2000,
                  n_jobs: int = 1,
                  verbose: bool = False,
                  use_gpu: bool = False) -> List[float]:
    """
    Evaluate multiple parameter configurations in parallel.
    
    Args:
        parameter_list: List of parameter arrays to evaluate
        train_timesteps: Number of training timesteps per evaluation
        eval_episodes: Number of evaluation episodes per configuration
        eval_timesteps: Maximum steps per evaluation episode
        n_jobs: Number of parallel jobs (currently not implemented)
        verbose: Whether to print progress information
        use_gpu: Whether to use GPU for training (default: False, uses CPU)
        
    Returns:
        List of average distances for each parameter configuration
    """
    results = []
    
    for i, parameters in enumerate(parameter_list):
        if verbose:
            print(f"[BATCH] Evaluating configuration {i+1}/{len(parameter_list)}")
        
        result = train_and_evaluate(
            parameters=parameters,
            train_timesteps=train_timesteps,
            eval_episodes=eval_episodes,
            eval_timesteps=eval_timesteps,
            seed=42 + i,  # Different seed for each evaluation
            verbose=verbose,
            use_gpu=use_gpu
        )
        
        results.append(result)
        
        if verbose:
            print(f"[BATCH] Configuration {i+1} result: {result:.3f}")
    
    return results


def test_baseline_performance(train_timesteps: int = 80000,
                             eval_episodes: int = 10,
                             eval_timesteps: int = 2000,
                            verbose: bool = True,
                            use_gpu: bool = False) -> float:
    """
    Test baseline performance with a simple reward function.
    
    This creates a baseline using weights that approximate the original reward:
    - Strong positive weight for forward_velocity
    - Negative weight for control_cost
    - Other features set to zero
    
    Args:
        train_timesteps: Number of training timesteps (default: 80000)
        eval_episodes: Number of evaluation episodes (default: 10)
        eval_timesteps: Maximum steps per evaluation episode (default: 2000)
        verbose: Whether to print progress information (default: True)
        use_gpu: Whether to use GPU for training (default: False, uses CPU)
    """
    if verbose:
        print("[BASELINE] Creating baseline reward function...")
        
    # Create a simple baseline reward approximating the original HalfCheetah reward
    try:
        # Load features to get the right dimension
        temp_reward = CanonicalReward(np.zeros(21))  # Default size
        feature_names = temp_reward.get_feature_names()
        n_features = len(feature_names)
        
        # Initialize all weights to zero
        baseline_weights = np.zeros(n_features)
        
        # Set key features based on canonical HalfCheetah reward
        for i, name in enumerate(feature_names):
            if name == "forward_velocity":
                baseline_weights[i] = 1.0  # Strong positive reward for forward motion
            elif name == "control_cost":
                baseline_weights[i] = -0.1  # Small penalty for control cost
            # Leave other features as zero
        
        if verbose:
            active_features = [(name, weight) for name, weight in zip(feature_names, baseline_weights) if abs(weight) > 1e-6]
            print(f"[BASELINE] Baseline uses {len(active_features)} features:")
            for name, weight in active_features:
                print(f"[BASELINE]   {name}: {weight:.3f}")
    
    except Exception as e:
        if verbose:
            print(f"[BASELINE] Warning: Could not create canonical baseline: {e}")
            print("[BASELINE] Using simple 2-feature baseline")
        
        # Fallback: create a minimal baseline
        baseline_weights = np.array([1.0, -0.1] + [0.0] * 19)  # Assume first two are velocity and control
        
    if verbose:
        print("[BASELINE] Testing baseline performance...")
    
    baseline_distance = train_and_evaluate(
        parameters=baseline_weights,
        train_timesteps=train_timesteps,
        eval_episodes=eval_episodes,
        eval_timesteps=eval_timesteps,
        seed=42,
        verbose=verbose,
        use_gpu=use_gpu
    )
    
    if verbose:
        print(f"[BASELINE] Baseline average distance: {baseline_distance:.3f}")
    
    return baseline_distance 