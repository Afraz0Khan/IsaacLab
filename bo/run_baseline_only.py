#!/usr/bin/env python3
"""
Run only the baseline performance test with the same configuration as the BO experiment.

Supports parallel evaluation over multiple fixed seeds.
"""

# Set threading limits BEFORE importing anything else
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")

import argparse
import multiprocessing as mp
import numpy as np
from pathlib import Path
from datetime import datetime

# Limit PyTorch threads after import
import torch
torch.set_num_threads(8)

from .train_evaluate import test_baseline_performance, train_and_evaluate
from .reward_function import CanonicalReward
import json
import matplotlib.pyplot as plt


def _get_canonical_feature_count_and_names():
    """Get the number of canonical features and their names from the feature file."""
    # Use the same feature loading logic as CanonicalReward
    from pathlib import Path
    
    # Search for feature file (same order as CanonicalReward._load_features)
    env_path = os.getenv("CANONICAL_FEATURES_FILE")
    if env_path and Path(env_path).exists():
        features_path = Path(env_path)
    else:
        current_dir = Path(__file__).parent
        candidates = [
            current_dir / "../starc_v2/scripts/seed_features.json",
            current_dir / "../starc_v2/scripts/features_simple.json",
            current_dir / "../../starc_v2/scripts/seed_features.json",
            current_dir / "../../starc_v2/scripts/features_simple.json",
            Path("starc_v2/scripts/seed_features.json"),
            Path("starc_v2/scripts/features_simple.json"),
        ]
        
        features_path = None
        for candidate in candidates:
            if candidate.exists():
                features_path = candidate
                break
                
        if features_path is None:
            # Fallback to a reasonable default
            return 25, [f"feature_{i}" for i in range(25)]
    
    try:
        with open(features_path, 'r') as f:
            features = json.load(f)
        feature_names = list(features.keys())
        return len(feature_names), feature_names
    except Exception:
        # Fallback if loading fails
        return 25, [f"feature_{i}" for i in range(25)]


def _baseline_worker(args):
    seed, train_timesteps, eval_episodes, eval_timesteps, verbose, baseline_weights = args
    return train_and_evaluate(
        parameters=baseline_weights,
        train_timesteps=train_timesteps,
        eval_episodes=eval_episodes,
        eval_timesteps=eval_timesteps,
        seed=seed,
        verbose=verbose,
        use_gpu=False,
    )


def main():
    """Run baseline performance test only."""
    parser = argparse.ArgumentParser(
        description='Run baseline performance test only',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Environment parameters  
    parser.add_argument('--train-timesteps', type=int, default=500000,
                        help='Training timesteps')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Evaluation episodes')
    parser.add_argument('--eval-timesteps', type=int, default=2000,
                        help='Steps per evaluation episode')
    parser.add_argument('--n-seeds', type=int, default=5,
                        help='Number of fixed seeds to average over (uses seeds 0..N-1)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    parser.add_argument('--results-dir', type=str, default='bo_results',
                        help='Directory to save baseline results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🎯 BASELINE PERFORMANCE TEST")
    print("="*80)
    print(f"Training: {args.train_timesteps} timesteps")
    print(f"Evaluation: {args.eval_episodes} episodes × {args.eval_timesteps} steps")
    seeds = list(range(max(1, args.n_seeds)))[:args.n_seeds]
    print(f"Seeds: {seeds}")
    print("-"*80)
    
    try:
        # Build baseline weights (forward_velocity=1.0, control_cost=-0.1)
        n_features, feature_names = _get_canonical_feature_count_and_names()
        
        baseline_weights = np.zeros(n_features, dtype=np.float32)
        # Prefer canonical names; fall back to synonyms
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        # Velocity feature
        if "x_velocity" in name_to_idx:
            baseline_weights[name_to_idx["x_velocity"]] = 1.0
        elif "forward_velocity" in name_to_idx:
            baseline_weights[name_to_idx["forward_velocity"]] = 1.0
        # Control cost feature
        if "ctrl_cost" in name_to_idx:
            baseline_weights[name_to_idx["ctrl_cost"]] = -0.1
        elif "control_cost" in name_to_idx:
            baseline_weights[name_to_idx["control_cost"]] = -0.1

        ctx = mp.get_context('spawn')
        worker_args = [
            (seed, args.train_timesteps, args.eval_episodes, args.eval_timesteps, (not args.quiet), baseline_weights)
            for seed in seeds
        ]
        with ctx.Pool(processes=min(len(seeds), os.cpu_count() or 1)) as pool:
            distances = pool.map(_baseline_worker, worker_args)

        baseline_distance = float(np.mean(distances)) if distances else 0.0
        baseline_std = float(np.std(distances)) if distances else 0.0
        baseline_min = float(np.min(distances)) if distances else 0.0
        baseline_max = float(np.max(distances)) if distances else 0.0

        # Save artifacts (JSON, TXT, plots) similar to BO outputs
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        out_dir = Path(args.results_dir) / f"baseline_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Results JSON
        results_json = {
            'type': 'baseline',
            'created_at': timestamp,
            'settings': {
                'train_timesteps': args.train_timesteps,
                'eval_episodes': args.eval_episodes,
                'eval_timesteps': args.eval_timesteps,
                'n_seeds': len(seeds)
            },
            'seeds': seeds,
            'distances': [float(d) for d in distances],
            'average_distance': baseline_distance,
            'std_distance': baseline_std,
            'min_distance': baseline_min,
            'max_distance': baseline_max,
            'feature_names': feature_names,
            'baseline_weights': [float(w) for w in baseline_weights]
        }
        with open(out_dir / 'baseline_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)

        # Text summary
        with open(out_dir / 'baseline_summary.txt', 'w') as f:
            f.write('BASELINE TEST SUMMARY\n')
            f.write('=' * 50 + '\n')
            f.write(f"Seeds: {seeds}\n")
            f.write(f"Per-seed distances: {[f'{d:.3f}' for d in distances]}\n")
            f.write(f"Average distance: {baseline_distance:.4f}\n")
            f.write(f"Std distance: {baseline_std:.4f}\n")
            f.write(f"Min/Max: {baseline_min:.3f} / {baseline_max:.3f}\n")
            f.write('\nBaseline weights (non-zeros):\n')
            for i, name in enumerate(feature_names):
                if abs(baseline_weights[i]) > 1e-9:
                    f.write(f"  {name}: {baseline_weights[i]:.3f}\n")

        # Plots
        try:
            # Histogram of distances
            plt.figure(figsize=(8, 5))
            plt.hist(distances, bins=max(5, len(distances)), alpha=0.8, color='steelblue', edgecolor='black')
            plt.axvline(baseline_distance, color='red', linestyle='--', label=f'Mean: {baseline_distance:.2f}')
            plt.title('Baseline Distance Distribution')
            plt.xlabel('Distance')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / 'baseline_distance_hist.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Per-seed bar chart
            plt.figure(figsize=(9, 5))
            x = np.arange(len(seeds))
            plt.bar(x, distances, color='teal', alpha=0.8)
            plt.xticks(x, [str(s) for s in seeds])
            plt.xlabel('Seed')
            plt.ylabel('Distance')
            plt.title('Baseline Distance per Seed')
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / 'baseline_per_seed.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception:
            pass
        
        print("\n" + "="*80)
        print("🏁 BASELINE TEST COMPLETE!")
        print("="*80)
        print(f"Per-seed distances: {[f'{d:.3f}' for d in distances]}")
        print(f"Average baseline distance (over {len(seeds)} seeds): {baseline_distance:.4f}")
        print(f"Artifacts saved to: {out_dir}")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n🛑 Baseline test interrupted by user")
    except Exception as e:
        print(f"\n❌ Baseline test failed: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main() 