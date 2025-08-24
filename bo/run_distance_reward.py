#!/usr/bin/env python3
"""
Run HalfCheetah with a distance-based reward over fixed seeds [0, 1, 2, 3, 4].

This script mirrors the baseline runner but replaces the reward function with
"distance travelled" (implemented as step reward = x_velocity). Distances are
measured robustly using MuJoCo qpos[0] deltas stored in the wrapper.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")

import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Tuple

import numpy as np

import torch
torch.set_num_threads(8)

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from .train_evaluate import make_halfcheetah_env, RewardWrapper


class DistanceReward:
    """Callable reward that returns x_velocity as per-step reward.

    RewardWrapper computes x_velocity from MuJoCo state; we simply return it.
    Total distance is measured independently from start/end x positions.
    """

    def __call__(self, env, state, action, next_state, x_velocity) -> float:
        try:
            return float(x_velocity)
        except Exception:
            return 0.0


def _distance_worker(args) -> float:
    seed, train_timesteps, eval_episodes, eval_timesteps, verbose = args

    # Build distance-based reward and environments
    reward_fn = DistanceReward()

    # Training env
    train_env = make_halfcheetah_env(max_episode_steps=eval_timesteps)
    train_env = RewardWrapper(train_env, reward_fn)
    train_env = Monitor(train_env)

    # Evaluation env
    eval_env = make_halfcheetah_env(max_episode_steps=eval_timesteps)
    eval_wrapper = RewardWrapper(eval_env, DistanceReward())
    eval_env = Monitor(eval_wrapper)

    try:
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0 if not verbose else 1,
            seed=seed,
            n_steps=2048,
            batch_size=64,
            learning_rate=3e-4,
            device="cpu",
        )

        # Train
        model.learn(total_timesteps=train_timesteps, progress_bar=verbose)

        # Clear distances collected during training evals
        eval_wrapper.clear_distances()

        # Final evaluation
        evaluate_policy(
            model, eval_env, n_eval_episodes=eval_episodes, deterministic=True, return_episode_rewards=True
        )

        # Use wrapper distances (true delta x)
        episode_distances = eval_wrapper.get_episode_distances()
        avg_distance = float(np.mean(episode_distances)) if len(episode_distances) > 0 else 0.0

        del model
        return avg_distance
    except Exception:
        try:
            del model  # type: ignore[name-defined]
        except Exception:
            pass
        return -1000.0
    finally:
        try:
            train_env.close()
            eval_env.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HalfCheetah with distance-based reward over fixed seeds [0..4]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--train-timesteps", type=int, default=300_000, help="Training timesteps per seed")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes per seed")
    parser.add_argument("--eval-timesteps", type=int, default=2000, help="Max steps per evaluation episode")
    parser.add_argument("--results-dir", type=str, default="bo_results", help="Directory to save results")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging")

    args = parser.parse_args()

    seeds = [0, 1, 2, 3, 4]
    verbose = (not args.quiet)

    print("=" * 80)
    print("🏃 HalfCheetah with Distance-Based Reward")
    print("=" * 80)
    print(f"Training: {args.train_timesteps} timesteps | Eval: {args.eval_episodes} × {args.eval_timesteps} steps")
    print(f"Seeds: {seeds}")
    print("-" * 80)

    ctx = mp.get_context("spawn")
    worker_args = [
        (seed, args.train_timesteps, args.eval_episodes, args.eval_timesteps, verbose) for seed in seeds
    ]
    with ctx.Pool(processes=min(len(seeds), os.cpu_count() or 1)) as pool:
        distances = pool.map(_distance_worker, worker_args)

    avg_distance = float(np.mean(distances)) if distances else 0.0
    std_distance = float(np.std(distances)) if distances else 0.0

    # Save artifacts
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.results_dir) / f"distance_reward_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    import json
    results = {
        "type": "distance_reward",
        "created_at": timestamp,
        "settings": {
            "train_timesteps": args.train_timesteps,
            "eval_episodes": args.eval_episodes,
            "eval_timesteps": args.eval_timesteps,
            "seeds": seeds,
        },
        "per_seed_distances": [float(d) for d in distances],
        "average_distance": avg_distance,
        "std_distance": std_distance,
    }
    with open(out_dir / "distance_reward_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # TXT
    with open(out_dir / "distance_reward_summary.txt", "w") as f:
        f.write("HALFCHEETAH – DISTANCE REWARD SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Per-seed distances: {[f'{d:.3f}' for d in distances]}\n")
        f.write(f"Average distance: {avg_distance:.4f}\n")
        f.write(f"Std distance: {std_distance:.4f}\n")

    # Plots (hist + per-seed bars)
    try:
        import matplotlib.pyplot as plt

        # Histogram
        plt.figure(figsize=(8, 5))
        plt.hist(distances, bins=max(5, len(distances)), alpha=0.8, color="steelblue", edgecolor="black")
        plt.axvline(avg_distance, color="red", linestyle="--", label=f"Mean: {avg_distance:.2f}")
        plt.title("Distance Reward – Distance Distribution")
        plt.xlabel("Distance")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "distance_reward_hist.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Per-seed bar chart
        plt.figure(figsize=(9, 5))
        x = np.arange(len(seeds))
        plt.bar(x, distances, color="teal", alpha=0.85)
        plt.xticks(x, [str(s) for s in seeds])
        plt.xlabel("Seed")
        plt.ylabel("Distance")
        plt.title("Distance Reward – Distance per Seed")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "distance_reward_per_seed.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    print("\n" + "=" * 80)
    print("✅ Distance reward run complete")
    print("Per-seed distances:", [f"{d:.3f}" for d in distances])
    print(f"Average distance (over {len(seeds)} seeds): {avg_distance:.4f}")
    print(f"Artifacts saved to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()


