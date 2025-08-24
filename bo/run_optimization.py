#!/usr/bin/env python3
"""
Main script to run Bayesian optimization for HalfCheetah canonical reward functions.

This script sets up and runs the complete BO pipeline with either random initialization
or STARC seed rewards as the initial population.
"""

# Set threading limits BEFORE importing anything else
import os
import multiprocessing as mp

"""Thread caps
Priority:
1) BO_THREADS_PER_PROCESS env var (integer)
2) Derived from CPU count and default n_eval_seeds=3
"""
cpu_count = mp.cpu_count()
env_threads = os.getenv("BO_THREADS_PER_PROCESS")
if env_threads and env_threads.isdigit():
    max_threads_per_process = max(1, int(env_threads))
else:
    n_eval_seeds = 3  # default planning assumption; actual used later
    max_threads_per_process = max(2, cpu_count // n_eval_seeds)

for var in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(var, str(max_threads_per_process))

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Limit PyTorch threads after import
import torch
torch.set_num_threads(max_threads_per_process)

from .objective import HalfCheetahObjective
from .bayesian_optimization import BayesianOptimizer
from .train_evaluate import test_baseline_performance
from .load_existing_rewards import get_seed_reward_info


def run_halfcheetah_bo(
    n_iterations: int = 50,
    n_initial_points: int = 10,
    train_timesteps: int = 80000,
    eval_episodes: int = 10,
    eval_timesteps: int = 2000,
    acquisition_function: str = 'UCB',
    exploration_factor: float = 2.0,
    device: str = 'auto',
    test_baseline: bool = False,
    results_dir: str = "bo_results",
    use_seed_rewards: bool = False,
    max_seed_rewards: int = None,
    n_eval_seeds: int = 3,
    batch_candidates: int = 1,
    bound_expansion: float = 2.0,
    initial_total_points: int = None,
    verbose: bool = True,
    discrete: bool = False,
    discrete_decimals: int = 2,
    fitness: str = "distance"
):
    """
    Run Bayesian optimization for HalfCheetah canonical reward functions.
    
    Args:
        n_iterations: Number of BO iterations after initial points
        n_initial_points: Number of Sobol initial points (ignored if use_seed_rewards=True)
        train_timesteps: Training timesteps for each evaluation
        eval_episodes: Number of evaluation episodes
        eval_timesteps: Steps per evaluation episode
        acquisition_function: 'UCB' or 'EI'
        exploration_factor: Exploration parameter (beta for UCB)
        device: PyTorch device ('cpu', 'cuda', or 'auto')
        test_baseline: Whether to test baseline performance first
        results_dir: Directory to save results
        use_seed_rewards: Whether to use STARC seed rewards as initial points
        max_seed_rewards: Maximum number of seed rewards to use (None for all)
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing optimization results
    """
    
    if verbose:
        print("="*80)
        print("🚀 BAYESIAN OPTIMIZATION FOR HALFCHEETAH CANONICAL REWARDS")
        print("="*80)
        print(f"Training: {train_timesteps} timesteps")
        print(f"Evaluation: {eval_episodes} episodes × {eval_timesteps} steps")
        print(f"System: {cpu_count} CPUs, {max_threads_per_process} threads/process")
        print(f"Multi-seed: {n_eval_seeds} parallel processes per evaluation")
        if batch_candidates > 1:
            print(f"⚠️  EXPERIMENTAL: Batch evaluation of {batch_candidates} candidates (may use significant memory)")
        
        if use_seed_rewards:
            seed_info = get_seed_reward_info()
            if seed_info['available']:
                actual_count = min(seed_info['count'], max_seed_rewards or seed_info['count'])
                print(f"Initial points: {actual_count} STARC seed rewards")
                if max_seed_rewards and max_seed_rewards < seed_info['count']:
                    print(f"  (using first {max_seed_rewards} of {seed_info['count']} available)")
                print(f"Weight bounds: Dynamic per feature (±{bound_expansion} from original STARC ranges)")
                if initial_total_points:
                    print(f"Augmenting to {initial_total_points} total initial points with Sobol sampling")
            else:
                print(f"Initial points: {n_initial_points} Sobol points (seed rewards not available)")
                print(f"Weight bounds: [-1, 1] for all features")
                use_seed_rewards = False
        else:
            print(f"Initial points: {n_initial_points} Sobol sequence")
            print(f"Weight bounds: [-1, 1] for all features")
            
        print(f"BO iterations: {n_iterations}")
        print(f"Acquisition: {acquisition_function} (factor={exploration_factor})")
        print(f"Results directory: {results_dir}")
        print("-"*80)
    
    # Set device configuration early
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    use_gpu = (device == 'cuda')
    
    # Baseline testing is optional and disabled by default
    baseline_distance = None
    if test_baseline:
        if verbose:
            print("\n[BASELINE] Testing baseline performance...")
        try:
            baseline_distance = test_baseline_performance(
                train_timesteps=train_timesteps,
                eval_episodes=eval_episodes,
                eval_timesteps=eval_timesteps,
                verbose=verbose,
                use_gpu=use_gpu
            )
            if verbose:
                print(f"[BASELINE] ✓ Baseline distance: {baseline_distance:.4f}")
        except Exception as e:
            if verbose:
                print(f"[BASELINE] ⚠️  Baseline test failed: {e}")
    
    # Create objective function
    if verbose:
        print("\n[SETUP] Creating optimization objective...")
        print(f"[SETUP] Using device: {device} (use_gpu={use_gpu})")
    
    objective = HalfCheetahObjective(
        train_timesteps=train_timesteps,
        eval_episodes=eval_episodes,
        eval_timesteps=eval_timesteps,
        n_eval_seeds=5 if n_eval_seeds is None else n_eval_seeds,
        verbose=verbose,
        use_gpu=use_gpu,
        fixed_seeds=[0,1,2,3,4],
        # Disable per-seed vectorization when using multi-seed parallelism by default
        n_envs=max(1, int(os.getenv("BO_VEC_ENVS", "1"))) if (n_eval_seeds or 1) <= 1 else 1,
        fitness_metric=fitness
    )
    
    if verbose:
        print(f"[SETUP] ✓ Objective created with {objective.get_feature_count()} features")
    
    # Initialize Bayesian optimizer
    if verbose:
        print(f"[SETUP] Initializing Bayesian optimizer on {device}...")
    
    optimizer = BayesianOptimizer(
        objective=objective,
        # Use all available seed rewards as initial points when requested
        n_initial_points=max_seed_rewards or n_initial_points,
        acquisition_function=acquisition_function,
        exploration_factor=exploration_factor,
        device=device,
        results_dir=results_dir,
        use_seed_rewards=use_seed_rewards,
        max_seed_rewards=max_seed_rewards,
        bound_expansion=bound_expansion,
        # If initial_total_points is not specified, and seed rewards are used,
        # set it to max_seed_rewards so we evaluate all seeds first.
        initial_total_points=(initial_total_points if initial_total_points is not None else (max_seed_rewards if use_seed_rewards else None)),
        discrete=discrete,
        discrete_decimals=discrete_decimals
    )
    
    # Run optimization
    if verbose:
        print("\n[OPTIMIZE] Starting Bayesian optimization...")
    
    results = optimizer.run_optimization(n_iterations=n_iterations)
    
    # Add baseline to results if available
    if baseline_distance is not None:
        results['baseline_distance'] = baseline_distance
        improvement = results['best_value'] - baseline_distance
        results['improvement_over_baseline'] = improvement
        if verbose:
            print(f"\n[RESULTS] Improvement over baseline: {improvement:+.4f}")
    
    # Final summary
    if verbose:
        print("\n" + "="*80)
        print("🏁 OPTIMIZATION COMPLETE!")
        print("="*80)
        print(f"Best distance found: {results['best_value']:.4f}")
        print(f"Total evaluations: {results['total_evaluations']}")
        print(f"Optimization time: {results['optimization_time']:.1f}s")
        print(f"Active features: {len([w for w in results['best_weights'] if abs(w) > 1e-6])}")
        
        # Show initialization method
        if results.get('use_seed_rewards', False):
            print(f"Initialization: {len(results.get('seed_reward_names', []))} STARC seed rewards")
        else:
            print(f"Initialization: {results['n_initial_points']} Sobol points")
        
        # Show top features
        feature_names = list(objective.get_feature_names())
        active_features = [(name, weight) for name, weight in zip(feature_names, results['best_weights']) 
                          if abs(weight) > 1e-6]
        active_features.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\nTop features by weight magnitude:")
        for name, weight in active_features[:5]:
            print(f"  {name}: {weight:+.4f}")
        
        print("="*80)
    
    return results


def main():
    """Command-line interface for BO optimization."""
    parser = argparse.ArgumentParser(
        description='Run Bayesian Optimization for HalfCheetah Canonical Rewards',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Optimization parameters
    parser.add_argument('--n-iterations', type=int, default=300,
                        help='Number of BO iterations')
    parser.add_argument('--n-initial', type=int, default=10,
                        help='Number of initial Sobol points (ignored if --use-seed-rewards)')
    parser.add_argument('--acquisition', choices=['UCB', 'EI'], default='EI',
                        help='Acquisition function')
    parser.add_argument('--exploration-factor', type=float, default=1.0,
                        help='Exploration factor (beta for UCB, ignored for EI)')
    
    # Environment parameters  
    parser.add_argument('--train-timesteps', type=int, default=80000,
                        help='Training timesteps per evaluation')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Evaluation episodes')
    parser.add_argument('--eval-timesteps', type=int, default=5000,
                        help='Steps per evaluation episode')
    
    # System parameters
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                        help='PyTorch device')
    parser.add_argument('--results-dir', type=str, default='bo_results',
                        help='Results directory')
    
    # Initialization options
    parser.add_argument('--use-seed-rewards', action='store_true',
                        help='Use STARC seed rewards as initial population instead of random points')
    parser.add_argument('--max-seed-rewards', type=int, default=None,
                        help='Maximum number of seed rewards to use (default: all available)')
    parser.add_argument('--n-eval-seeds', type=int, default=3,
                        help='Number of random seeds to average per evaluation (default: 3)')
    parser.add_argument('--starc-env', type=str, default=None,
                        help='Environment name to read STARC seeds from (halfcheetah, ant, humanoid)')
    
    # Options
    parser.add_argument('--baseline', action='store_true',
                        help='Run baseline performance test before optimization')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    
    # Quick test mode
    parser.add_argument('--quick-test', action='store_true',
                        help='Run with reduced parameters for quick testing')
    
    # Performance optimizations
    parser.add_argument('--batch-candidates', type=int, default=1,
                        help='Number of BO candidates to evaluate in parallel (experimental, default: 1)')
    parser.add_argument('--reduce-seeds', type=int, default=None,
                        help='Reduce number of evaluation seeds for faster iteration (default: 3)')
    parser.add_argument('--bound-expansion', type=float, default=2.0,
                        help='How much to expand BO bounds beyond STARC weight ranges (default: 2.0)')
    parser.add_argument('--initial-total-points', type=int, default=None,
                        help='If using seed rewards, augment with Sobol points to reach this many initial points')
    parser.add_argument('--discrete', '-discrete', action='store_true',
                        help='Enable discrete search: round all weights to a fixed decimal precision grid')
    parser.add_argument('--discrete-decimals', type=int, default=2,
                        help='Number of decimal places to keep when --discrete is enabled (e.g., 2 -> step 0.01)')
    parser.add_argument('--fitness', choices=['distance', 'expert'], default='distance',
                        help='Fitness metric to optimize: distance traveled or expert reward')
    
    # Special modes
    parser.add_argument('--plot-seed-rewards', action='store_true',
                        help='Plot STARC seed rewards and exit')
    parser.add_argument('--seed-info', action='store_true',
                        help='Show information about available seed rewards and exit')
    
    args = parser.parse_args()
    
    # Handle special modes
    if args.seed_info:
        print("STARC Seed Reward Information")
        print("="*40)
        seed_info = get_seed_reward_info()
        if seed_info['available']:
            print(f"✓ Found {seed_info['count']} seed reward functions")
            print(f"✓ Canonical features: {seed_info['n_canonical_features']}")
            print(f"\nSeed reward files:")
            for i, path in enumerate(seed_info['paths'], 1):
                print(f"  {i:2d}. {Path(path).stem}")
        else:
            print(f"✗ Seed rewards not available: {seed_info['error']}")
        return
    
    if args.plot_seed_rewards:
        print("Plotting STARC seed reward functions...")
        try:
            from .load_existing_rewards import load_seed_rewards_as_initial_points, load_canonical_features, plot_seed_rewards
            
            initial_points, reward_names, _ = load_seed_rewards_as_initial_points()
            canonical_features = load_canonical_features()
            
            plot_seed_rewards(
                initial_points, 
                reward_names, 
                canonical_features,
                save_path="seed_rewards_visualization.png"
            )
            print("✓ Plot saved to seed_rewards_visualization.png")
            
        except Exception as e:
            print(f"✗ Failed to plot seed rewards: {e}")
        return
    
    # Quick test mode adjustments
    if args.quick_test:
        args.n_iterations = 5
        args.n_initial = 3
        args.train_timesteps = 10000
        args.eval_episodes = 3
        args.eval_timesteps = 500
        print("[QUICK TEST MODE] Using reduced parameters for fast testing")
    
    # Apply seed reduction if specified
    if args.reduce_seeds is not None:
        args.n_eval_seeds = args.reduce_seeds
        print(f"[OPTIMIZATION] Reducing evaluation seeds to {args.n_eval_seeds} for faster iteration")
    
    # Propagate env selection for STARC seeds (new namespaced layout)
    if args.starc_env:
        os.environ["STARCV2_ENV"] = args.starc_env
    
    try:
        results = run_halfcheetah_bo(
            n_iterations=args.n_iterations,
            n_initial_points=args.n_initial,
            train_timesteps=args.train_timesteps,
            eval_episodes=args.eval_episodes,
            eval_timesteps=args.eval_timesteps,
            acquisition_function=args.acquisition,
            exploration_factor=args.exploration_factor,
            device=args.device,
            test_baseline=args.baseline,
            results_dir=args.results_dir,
            use_seed_rewards=args.use_seed_rewards,
            max_seed_rewards=args.max_seed_rewards,
            n_eval_seeds=args.n_eval_seeds,
            batch_candidates=args.batch_candidates,
            bound_expansion=args.bound_expansion,
            initial_total_points=args.initial_total_points,
            verbose=not args.quiet,
            discrete=args.discrete,
            discrete_decimals=args.discrete_decimals,
            fitness=args.fitness
        )
        
        print(f"\n✅ Optimization completed successfully!")
        print(f"Results saved to: {Path(args.results_dir).absolute()}")
        
        # Show key results
        if not args.quiet:
            print(f"\n📊 Key Results:")
            print(f"Best distance: {results['best_value']:.4f}")
            print(f"Total evaluations: {results['total_evaluations']}")
            if results.get('use_seed_rewards'):
                print(f"Used {len(results.get('seed_reward_names', []))} STARC seed rewards as initial population")
        
    except KeyboardInterrupt:
        print("\n🛑 Optimization interrupted by user")
        print("Results have been saved incrementally during the run.")
        print(f"Check the '{args.results_dir}' directory for:")
        print("  - bo_latest_checkpoint.json (latest state)")
        print("  - bo_intermediate_iter*.json (periodic saves)")
        print("  - seed_rewards_initial_population.png (if using seed rewards)")
        sys.exit(0)  # Exit successfully since we saved results
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 


