#!/usr/bin/env python3
"""
compare_best_vs_baseline.py
--------------------------------
A utility script to run a **long-horizon** comparison between the best
reward found by Bayesian optimisation (stored in *bo_latest_checkpoint.json*)
and the canonical baseline reward.

The script:
1. Reads the checkpoint JSON (either an intermediate checkpoint produced
   during optimisation or a final results file) and pulls out the best
   weight vector.
2. Trains a PPO agent with this reward for the specified number of
   **train_timesteps** (default: 1_000_000) and then evaluates it.
3. Trains a baseline agent with the same budget and evaluates it.
4. Prints the two average distances and the improvement gap.

Example
-------
python -m bo.compare_best_vs_baseline \
    --checkpoint bo_results/bo_latest_checkpoint.json \
    --train-timesteps 1000000 \
    --eval-episodes 10 \
    --eval-timesteps 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Re-use existing evaluation utilities
from .train_evaluate import train_and_evaluate, test_baseline_performance

# ---------------------------------------------------------------------------
# Helper: train agent and record reward-vs-timesteps curve
# ---------------------------------------------------------------------------

def train_and_collect_rewards(
    parameters: np.ndarray | None,
    train_timesteps: int,
    eval_freq: int = 10_000,
    n_eval_episodes: int = 3,
    eval_timesteps: int = 1_000,
    seed: int | None = None,
    verbose: bool = False,
    log_dir: Path | None = None,
    n_seeds: int = 1,  # Number of different seeds to average over
    hybrid_training: bool = False,
    switch_timesteps: int = 50000,
    expert_parameters: np.ndarray | None = None,
):
    """Train PPO with evaluation callback that writes *evaluations.npz* and return curve.
    
    If n_seeds > 1, runs multiple training sessions with different seeds and averages results.

    Returns (distance, timesteps_array, avg_rewards_array, avg_distances_array).
    """

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.evaluation import evaluate_policy

    # Local imports to avoid circular deps
    from .train_evaluate import make_halfcheetah_env, RewardWrapper
    from .reward_function import CanonicalReward

    # Prepare logging directory
    if log_dir is None:
        log_dir = Path(".") / "tmp_logs"
    log_dir = Path(log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build reward fn and environments
    if parameters is None:
        # Create baseline reward (forward_velocity=1.0, control_cost=-0.1)
        from .train_evaluate import test_baseline_performance
        temp_reward = CanonicalReward(np.zeros(21))  # Get feature count
        feature_names = temp_reward.get_feature_names()
        baseline_weights = np.zeros(len(feature_names))
        for i, name in enumerate(feature_names):
            if name == "forward_velocity":
                baseline_weights[i] = 1.0
            elif name == "control_cost":
                baseline_weights[i] = -0.1
        parameters = baseline_weights
    
    reward_fn = CanonicalReward(parameters)
    train_env = make_halfcheetah_env(max_episode_steps=eval_timesteps)
    train_env = RewardWrapper(train_env, reward_fn)

    eval_env = make_halfcheetah_env(max_episode_steps=eval_timesteps)
    eval_env = RewardWrapper(eval_env, CanonicalReward(parameters))
    from stable_baselines3.common.monitor import Monitor
    eval_env = Monitor(eval_env)

    # Custom callback to track distances during training
    class DistanceTrackingCallback:
        def __init__(self, eval_env, eval_freq, n_eval_episodes, log_dir):
            self.eval_env = eval_env
            self.eval_freq = eval_freq  
            self.n_eval_episodes = n_eval_episodes
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Track distances and rewards over time
            self.timesteps = []
            self.distances = []
            self.rewards = []
            
        def __call__(self, locals_, globals_):
            # Check if it's time to evaluate
            if locals_['self'].num_timesteps % self.eval_freq == 0:
                model = locals_['self']
                
                # Clear previous distances
                if hasattr(self.eval_env, 'clear_distances'):
                    self.eval_env.clear_distances()
                elif hasattr(self.eval_env, 'env') and hasattr(self.eval_env.env, 'clear_distances'):
                    self.eval_env.env.clear_distances()
                
                # Evaluate model
                from stable_baselines3.common.evaluation import evaluate_policy
                episode_rewards, episode_lengths = evaluate_policy(
                    model, self.eval_env, n_eval_episodes=self.n_eval_episodes,
                    deterministic=True, return_episode_rewards=True
                )
                
                # Get distances
                distances = []
                if hasattr(self.eval_env, 'get_episode_distances'):
                    distances = self.eval_env.get_episode_distances()
                elif hasattr(self.eval_env, 'env') and hasattr(self.eval_env.env, 'get_episode_distances'):
                    distances = self.eval_env.env.get_episode_distances()
                
                # Store results
                self.timesteps.append(model.num_timesteps)
                self.rewards.append(float(np.mean(episode_rewards)))
                self.distances.append(float(np.mean(distances)) if distances else 0.0)
                
                if verbose:
                    print(f"[EVAL] Step {model.num_timesteps}: Reward={self.rewards[-1]:.1f}, Distance={self.distances[-1]:.1f}")
                
                # Save to npz file (compatible format)
                np.savez(
                    self.log_dir / "evaluations.npz",
                    timesteps=np.array(self.timesteps),
                    results=np.array([[r] * self.n_eval_episodes for r in self.rewards]),  # Fake multiple episodes for compatibility
                    distances=np.array(self.distances),
                    ep_lengths=np.array([[1000] * self.n_eval_episodes for _ in self.rewards])
                )
            
            return True
    
    distance_callback = DistanceTrackingCallback(eval_env, eval_freq, n_eval_episodes, log_dir)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0 if not verbose else 1,
        seed=seed,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        device="cpu",  # Force CPU for consistency with BO experiment
    )

    if hybrid_training and expert_parameters is not None and switch_timesteps < train_timesteps:
        # Phase 1: Train with BO reward for switch_timesteps
        if verbose:
            print(f"[HYBRID] Phase 1: Training with BO reward for {switch_timesteps} timesteps...")
        model.learn(total_timesteps=switch_timesteps, callback=distance_callback, progress_bar=verbose)
        
        # Phase 2: Switch to expert reward and continue training
        remaining_timesteps = train_timesteps - switch_timesteps
        if verbose:
            print(f"[HYBRID] Phase 2: Switching to expert reward for remaining {remaining_timesteps} timesteps...")
        
        # Create new environment with expert reward
        expert_reward_fn = CanonicalReward(expert_parameters)
        new_train_env = make_halfcheetah_env(max_episode_steps=eval_timesteps)
        new_train_env = RewardWrapper(new_train_env, expert_reward_fn)
        
        # Update model's environment
        model.set_env(new_train_env)
        
        # Continue training with expert reward
        model.learn(total_timesteps=remaining_timesteps, callback=distance_callback, 
                   progress_bar=verbose, reset_num_timesteps=False)
        
        # Close the new environment
        new_train_env.close()
    else:
        # Standard training
        model.learn(total_timesteps=train_timesteps, callback=distance_callback, progress_bar=verbose)

    # Load evaluation curve
    eval_file = Path(log_dir) / "evaluations.npz"
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

    data = np.load(eval_file)
    timesteps_arr: np.ndarray = data["timesteps"].flatten()
    results_arr: np.ndarray = data["results"]  # shape (n_eval, n_episodes)
    avg_rewards_arr = results_arr.mean(axis=1)
    
    # Load distances if available
    distances_arr = data.get("distances", np.zeros_like(timesteps_arr))

    # Evaluate the TRAINED model directly for distance (no retraining!)
    if verbose:
        print(f"[EVAL] Evaluating trained model over {n_eval_episodes} episodes for distance...")
    
    # Create fresh eval environment to measure distances
    from stable_baselines3.common.monitor import Monitor
    distance_eval_env = make_halfcheetah_env(max_episode_steps=eval_timesteps)
    distance_reward_wrapper = RewardWrapper(distance_eval_env, CanonicalReward(parameters))
    distance_eval_env = Monitor(distance_reward_wrapper)
    
    # Clear any previous distances and evaluate the trained model
    distance_reward_wrapper.clear_distances()
    
    from stable_baselines3.common.evaluation import evaluate_policy
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        distance_eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        return_episode_rewards=True
    )
    
    # Get distances from the wrapper
    episode_distances = distance_reward_wrapper.get_episode_distances()
    
    if len(episode_distances) > 0:
        distance = float(np.mean(episode_distances))
    else:
        distance = 0.0
        
    if verbose:
        print(f"[EVAL] Average distance from trained model: {distance:.3f}")
    
    distance_eval_env.close()

    # Clean up envs
    train_env.close()
    eval_env.close()

    return distance, timesteps_arr, avg_rewards_arr, distances_arr


def load_best_weights(checkpoint_path: Path) -> np.ndarray:
    """Extract the best weight vector from a BO checkpoint or results file."""
    with open(checkpoint_path, "r") as f:
        data: dict[str, Any] = json.load(f)

    # Support both intermediate checkpoint and final results structures
    if "current_best_weights" in data:
        weights = data["current_best_weights"]
    elif "best_weights" in data:
        weights = data["best_weights"]
    else:
        raise KeyError(
            "Could not find 'current_best_weights' or 'best_weights' in the provided JSON."
        )

    return np.asarray(weights, dtype=float)


def _parallel_comparison_worker(args):
    """
    Worker function for parallel comparison of baseline vs best reward.
    This needs to be at module level to be picklable by multiprocessing.
    """
    (reward_type, parameters, train_timesteps, eval_episodes, eval_timesteps, 
     eval_freq, seed, verbose, log_dir, hybrid_training, switch_timesteps, expert_parameters) = args
    
    return train_and_collect_rewards(
        parameters=parameters,
        train_timesteps=train_timesteps,
        n_eval_episodes=eval_episodes,
        eval_timesteps=eval_timesteps,
        eval_freq=eval_freq,
        seed=seed,
        verbose=verbose,
        log_dir=log_dir,
        hybrid_training=hybrid_training,
        switch_timesteps=switch_timesteps,
        expert_parameters=expert_parameters
    )


def run_parallel_comparison(best_weights: np.ndarray, train_timesteps: int, 
                          eval_episodes: int, eval_timesteps: int, eval_freq: int,
                          n_seeds: int, verbose: bool, output_dir: str,
                          hybrid_training: bool = False, switch_timesteps: int = 50000,
                          sequential: bool = True, baseline_first: bool = True):
    """
    Run baseline vs best reward comparison in parallel.
    
    Returns:
        Tuple of (best_results, baseline_results) where each contains
        (distance, timesteps_array, avg_rewards_array, avg_distances_array)
    """
    import multiprocessing as mp
    import tempfile
    import os
    
    # Set fixed random seeds for reproducibility (match BO's 5 seeds 0..4)
    fixed_seeds = [0, 1, 2, 3, 4]
    if n_seeds > len(fixed_seeds):
        print(f"[WARNING] Requested {n_seeds} seeds but only {len(fixed_seeds)} fixed seeds available. Using first {len(fixed_seeds)}.")
        n_seeds = len(fixed_seeds)
    
    print(f"[COMPARISON] Running parallel comparison with {n_seeds} fixed seeds: {fixed_seeds[:n_seeds]}")
    print(f"[COMPARISON] Best reward: {np.sum(np.abs(best_weights) > 1e-6)} active features")
    print(f"[COMPARISON] Baseline reward: forward_velocity=1.0, control_cost=-0.1")
    
    # Create baseline weights mapped to canonical feature names (supports both naming schemes)
    def _resolve_feature_names() -> list[str]:
        # Try to read from canonical feature json; otherwise default to length of best_weights
        from pathlib import Path
        import os as _os
        env_path = _os.getenv("CANONICAL_FEATURES_FILE")
        candidates = []
        if env_path and Path(env_path).exists():
            candidates.append(Path(env_path))
        current_dir = Path(__file__).parent
        candidates += [
            current_dir / "../starc_v2/scripts/seed_features.json",
            current_dir / "../../starc_v2/scripts/seed_features.json",
            Path("starc_v2/scripts/seed_features.json"),
            Path("../starc_v2/scripts/seed_features.json"),
            current_dir / "../starc_v2/scripts/features_simple.json",
            current_dir / "../../starc_v2/scripts/features_simple.json",
            Path("starc_v2/scripts/features_simple.json"),
            Path("../starc_v2/scripts/features_simple.json"),
        ]
        for p in candidates:
            try:
                if p.exists():
                    import json as _json
                    feats = _json.load(open(p, 'r'))
                    return list(feats.keys())
            except Exception:
                continue
        return [f"feature_{i}" for i in range(len(best_weights))]

    feature_names = _resolve_feature_names()
    baseline_weights = np.zeros(len(feature_names), dtype=float)
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    # velocity term
    if "x_velocity" in name_to_idx:
        baseline_weights[name_to_idx["x_velocity"]] = 1.0
    elif "forward_velocity" in name_to_idx:
        baseline_weights[name_to_idx["forward_velocity"]] = 1.0
    # control cost term
    if "ctrl_cost" in name_to_idx:
        baseline_weights[name_to_idx["ctrl_cost"]] = -0.1
    elif "control_cost" in name_to_idx:
        baseline_weights[name_to_idx["control_cost"]] = -0.1
    
    # Create expert reward weights (same as baseline for now)
    expert_weights = baseline_weights.copy()
    
    if hybrid_training:
        print(f"[COMPARISON] Using HYBRID training: BO reward for {switch_timesteps} steps, then expert reward")
        print(f"[COMPARISON] Expert reward: forward_velocity=1.0, control_cost=-0.1")
    
    # Create temporary directories for each seed
    temp_dirs = [tempfile.mkdtemp(prefix=f"comparison_seed{seed}_") for seed in fixed_seeds[:n_seeds]]

    def _run_group(kind: str) -> list:
        # kind: "best" or "baseline"
        args_list = []
        for seed, temp_dir in zip(fixed_seeds[:n_seeds], temp_dirs):
            if kind == "best":
                args_list.append((
                    "best", best_weights, train_timesteps, eval_episodes, eval_timesteps,
                    eval_freq, seed, verbose, temp_dir, hybrid_training, switch_timesteps, expert_weights
                ))
            else:
                args_list.append((
                    "baseline", baseline_weights, train_timesteps, eval_episodes, eval_timesteps,
                    eval_freq, seed, verbose, temp_dir, False, 0, None
                ))
        ctx = mp.get_context('spawn')
        # Use n_seeds processes per group
        with ctx.Pool(processes=min(n_seeds, os.cpu_count())) as pool:
            return pool.map(_parallel_comparison_worker, args_list)

    if sequential:
        order = ("baseline", "best") if baseline_first else ("best", "baseline")
        if order[0] == "baseline":
            baseline_results = _run_group("baseline")
            best_results = _run_group("best")
        else:
            best_results = _run_group("best")
            baseline_results = _run_group("baseline")
    else:
        # Fall back to previous behavior: interleaved best and baseline together
        worker_args = []
        for seed, temp_dir in zip(fixed_seeds[:n_seeds], temp_dirs):
            worker_args.append((
                "best", best_weights, train_timesteps, eval_episodes, eval_timesteps,
                eval_freq, seed, verbose, temp_dir, hybrid_training, switch_timesteps, expert_weights
            ))
            worker_args.append((
                "baseline", baseline_weights, train_timesteps, eval_episodes, eval_timesteps,
                eval_freq, seed, verbose, temp_dir, False, 0, None
            ))
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=min(n_seeds * 2, os.cpu_count())) as pool:
            results = pool.map(_parallel_comparison_worker, worker_args)
        # Separate best and baseline results
        best_results, baseline_results = [], []
        for i, result in enumerate(results):
            (best_results if i % 2 == 0 else baseline_results).append(result)
    
    # Clean up temp directories
    for temp_dir in temp_dirs:
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
    
    # Average results across seeds
    best_distance = np.mean([r[0] for r in best_results])
    baseline_distance = np.mean([r[0] for r in baseline_results])
    
    # Average timesteps and curves (assuming same timestep structure)
    best_timesteps = best_results[0][1]  # Use first result's timesteps
    baseline_timesteps = baseline_results[0][1]
    
    best_rewards = np.mean([r[2] for r in best_results], axis=0)
    baseline_rewards = np.mean([r[2] for r in baseline_results], axis=0)
    
    best_distances = np.mean([r[3] for r in best_results], axis=0)
    baseline_distances = np.mean([r[3] for r in baseline_results], axis=0)
    
    print(f"[COMPARISON] Results averaged over {n_seeds} fixed seeds: {fixed_seeds[:n_seeds]}")
    print(f"[COMPARISON] Best reward distance: {best_distance:.3f}")
    print(f"[COMPARISON] Baseline distance: {baseline_distance:.3f}")
    print(f"[COMPARISON] Improvement: {best_distance - baseline_distance:+.3f}")
    
    return (best_distance, best_timesteps, best_rewards, best_distances), \
           (baseline_distance, baseline_timesteps, baseline_rewards, baseline_distances), \
           (best_results, baseline_results)  # Return individual seed results too, 


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the best BO reward against the baseline on a long training horizon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to bo_latest_checkpoint.json or a final bo_results_*.json file.",
    )
    parser.add_argument(
        "--train-timesteps",
        type=int,
        default=1_000_000,
        help="Training timesteps for each agent.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes after training.",
    )
    parser.add_argument(
        "--eval-timesteps",
        type=int,
        default=5_000,
        help="Maximum steps per evaluation episode.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information from training/evaluation routines.",
    )

    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Frequency (in timesteps) of evaluation during training curves.",
    )
    
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Number of different random seeds to average over (for robustness).",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="compare_plots",
        help="Directory to save generated plots (created if needed).",
    )
    
    parser.add_argument(
        "--hybrid-training",
        action="store_true",
        help="Use hybrid training: BO reward for X steps, then switch to expert reward.",
    )
    
    parser.add_argument(
        "--switch-timesteps",
        type=int,
        default=50000,
        help="Timesteps after which to switch from BO reward to expert reward (default: 50000).",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        sys.exit(f"Checkpoint file not found: {checkpoint_path}")

    print("\n🔍 Loading checkpoint …")
    try:
        best_weights = load_best_weights(checkpoint_path)
    except Exception as e:
        sys.exit(f"Failed to load best weights: {e}")

    print(f"✓ Loaded best weight vector with {len(best_weights)} features")

    # Directory for logs
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Run PARALLEL comparison: BEST vs BASELINE
    # ------------------------------------------------------------------
    print("\n🚀 Running PARALLEL comparison: BEST vs BASELINE …")
    (best_distance, best_timesteps, best_rewards_curve, best_distances_curve), \
    (baseline_distance, base_timesteps, base_rewards_curve, base_distances_curve), \
    (best_seed_results, baseline_seed_results) = run_parallel_comparison(
        best_weights=best_weights,
        train_timesteps=args.train_timesteps,
        eval_episodes=args.eval_episodes,
        eval_timesteps=args.eval_timesteps,
        eval_freq=args.eval_freq,
        n_seeds=args.n_seeds,
        verbose=args.verbose,
        output_dir=str(output_dir),
        hybrid_training=args.hybrid_training,
        switch_timesteps=args.switch_timesteps
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    improvement = best_distance - baseline_distance
    seeds_used = [0, 1, 2, 3, 4][:args.n_seeds]
    print("\n============================================================")
    print(f"BEST reward distance   : {best_distance:.3f}")
    print(f"BASELINE distance      : {baseline_distance:.3f}")
    print(f"Improvement (BEST - BL): {improvement:+.3f}")
    print(f"Seeds used            : {seeds_used}")
    print("============================================================")

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------

    # 1) Distance comparison bar chart
    try:
        import matplotlib.pyplot as plt

        labels = ["Best Reward", "Baseline"]
        values = [best_distance, baseline_distance]
        colors = ["steelblue", "gray"]

        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, values, color=colors, alpha=0.8)
        plt.ylabel("Average Distance")
        training_type = f"Hybrid (BO→Expert @{args.switch_timesteps})" if args.hybrid_training else "BO-optimized"
        plt.title("Long-Horizon Performance ({} steps)\n{} vs Baseline | Averaged over {} seeds: {}".format(
            args.train_timesteps, training_type, args.n_seeds, seeds_used))
        plt.grid(axis="y", alpha=0.3)

        # Annotate bars
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02 * max(values), f"{val:.1f}",
                     ha="center", va="bottom", fontsize=10)

        distance_plot_path = output_dir / "best_vs_baseline_distance.png"
        plt.tight_layout()
        plt.savefig(distance_plot_path, dpi=300)
        plt.close()
        print(f"📊 Distance comparison plot saved to: {distance_plot_path}")
    except Exception as e:
        print(f"[WARN] Could not create distance comparison plot: {e}")

    # 2) Feature importance plot for BEST reward
    try:
        from .reward_function import CanonicalReward
        from .utils import plot_feature_importance

        # Get canonical feature names
        temp_reward = CanonicalReward(np.zeros_like(best_weights))
        feature_names = temp_reward.get_feature_names()

        importance_path = output_dir / "best_feature_importance.png"
        plot_feature_importance(feature_names, best_weights.tolist(), save_path=importance_path,
                                title="Best Reward – Feature Importance")
    except Exception as e:
        print(f"[WARN] Could not create feature importance plot: {e}")

    # 3) Reward-vs-timesteps curve
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 7))
        
        # Plot individual seed lines for Best reward
        for i, (distance, timesteps, rewards, distances) in enumerate(best_seed_results):
            alpha = 0.3 if i > 0 else 0.5  # First seed more visible
            plt.plot(timesteps, rewards, color="steelblue", alpha=alpha, linewidth=1, 
                    label=f"Best Seed {seeds_used[i]}" if i == 0 else "")
        
        # Plot individual seed lines for Baseline
        for i, (distance, timesteps, rewards, distances) in enumerate(baseline_seed_results):
            alpha = 0.3 if i > 0 else 0.5
            plt.plot(timesteps, rewards, color="gray", alpha=alpha, linewidth=1,
                    label=f"Baseline Seed {seeds_used[i]}" if i == 0 else "")
        
        # Plot averaged lines (thicker)
        plt.plot(best_timesteps, best_rewards_curve, label="Best Reward (Avg)", 
                color="steelblue", linewidth=3, alpha=0.8)
        plt.plot(base_timesteps, base_rewards_curve, label="Baseline (Avg)", 
                color="gray", linewidth=3, alpha=0.8)
        
        plt.xlabel("Timesteps")
        plt.ylabel("Average Episode Reward (eval)")
        plt.title("Training Progress – Average Reward vs Timesteps\nIndividual seeds + averaged over {} seeds: {}".format(
            args.n_seeds, seeds_used))
        plt.legend()
        plt.grid(alpha=0.3)

        reward_curve_path = output_dir / "reward_vs_timesteps.png"
        plt.tight_layout()
        plt.savefig(reward_curve_path, dpi=300)
        plt.close()
        print(f"📈 Reward-vs-timesteps plot saved to: {reward_curve_path}")
    except Exception as e:
        print(f"[WARN] Could not create reward-vs-timesteps plot: {e}")

    # 4) Distance-vs-timesteps curve
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 7))
        
        # Plot individual seed lines for Best reward
        for i, (distance, timesteps, rewards, distances) in enumerate(best_seed_results):
            alpha = 0.3 if i > 0 else 0.5
            plt.plot(timesteps, distances, color="steelblue", alpha=alpha, linewidth=1,
                    label=f"Best Seed {seeds_used[i]}" if i == 0 else "")
        
        # Plot individual seed lines for Baseline
        for i, (distance, timesteps, rewards, distances) in enumerate(baseline_seed_results):
            alpha = 0.3 if i > 0 else 0.5
            plt.plot(timesteps, distances, color="gray", alpha=alpha, linewidth=1,
                    label=f"Baseline Seed {seeds_used[i]}" if i == 0 else "")
        
        # Plot averaged lines (thicker)
        plt.plot(best_timesteps, best_distances_curve, label="Best Reward (Avg)", 
                color="steelblue", linewidth=3, alpha=0.8)
        plt.plot(base_timesteps, base_distances_curve, label="Baseline (Avg)", 
                color="gray", linewidth=3, alpha=0.8)
        
        plt.xlabel("Timesteps")
        plt.ylabel("Average Distance Traveled (eval)")
        plt.title("Training Progress – Average Distance vs Timesteps\nIndividual seeds + averaged over {} seeds: {}".format(
            args.n_seeds, seeds_used))
        plt.legend()
        plt.grid(alpha=0.3)

        distance_curve_path = output_dir / "distance_vs_timesteps.png"
        plt.tight_layout()
        plt.savefig(distance_curve_path, dpi=300)
        plt.close()
        print(f"📏 Distance-vs-timesteps plot saved to: {distance_curve_path}")
    except Exception as e:
        print(f"[WARN] Could not create distance-vs-timesteps plot: {e}")

    print("\nPlots saved to:", output_dir, "\n")


if __name__ == "__main__":
    main() 