#!/usr/bin/env python3
"""
Bayesian Optimization runner for IsaacLab environments (Ant and Humanoid).

This script runs BO to discover optimal canonical reward functions for
IsaacLab tasks using RL-Games training with 300k timesteps per evaluation.
"""

import argparse
import torch
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from .isaaclab_objective import create_isaaclab_objective
from .bayesian_optimization import BayesianOptimizer
from .utils import save_results, plot_convergence, plot_feature_importance


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Bayesian Optimization for IsaacLab Canonical Rewards"
    )
    
    # Environment selection
    parser.add_argument(
        "--task", 
        choices=["ant", "humanoid"], 
        required=True,
        help="IsaacLab task to optimize (ant or humanoid)"
    )
    
    # BO parameters
    parser.add_argument(
        "--n-iterations", 
        type=int, 
        default=50,
        help="Number of BO iterations after initial points (default: 50)"
    )
    parser.add_argument(
        "--n-initial-points", 
        type=int, 
        default=10,
        help="Number of initial Sobol points for exploration (default: 10)"
    )
    parser.add_argument(
        "--acquisition", 
        choices=["UCB", "EI", "PI"], 
        default="UCB",
        help="Acquisition function (default: UCB)"
    )
    parser.add_argument(
        "--exploration-factor", 
        type=float, 
        default=2.0,
        help="Exploration parameter for UCB (β) or EI (default: 2.0)"
    )
    
    # Training parameters
    parser.add_argument(
        "--train-timesteps", 
        type=int, 
        default=300000,
        help="Training timesteps per evaluation (default: 300k)"
    )
    parser.add_argument(
        "--eval-steps", 
        type=int, 
        default=2000,
        help="Evaluation steps per episode (default: 2000)"
    )
    parser.add_argument(
        "--eval-num-envs", 
        type=int, 
        default=16,
        help="Number of parallel evaluation environments (default: 16)"
    )
    parser.add_argument(
        "--n-eval-seeds", 
        type=int, 
        default=5,
        help="Number of seeds to average per evaluation (default: 5)"
    )
    
    # System parameters
    parser.add_argument(
        "--device", 
        choices=["cpu", "cuda", "auto"], 
        default="auto",
        help="Device for BoTorch optimization (default: auto)"
    )
    parser.add_argument(
        "--isaaclab-path", 
        type=str, 
        default="/home/afraz1/Documents/IsaacLab",
        help="Path to IsaacLab installation"
    )
    parser.add_argument(
        "--results-dir", 
        type=str, 
        default="bo_results_isaaclab",
        help="Directory to save results (default: bo_results_isaaclab)"
    )
    
    # Testing and debugging
    parser.add_argument(
        "--quick-test", 
        action="store_true",
        help="Run with reduced parameters for quick testing"
    )
    parser.add_argument(
        "--test-objective", 
        action="store_true",
        help="Test objective function with random points and exit"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable detailed progress output"
    )
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup PyTorch device for BoTorch."""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    print(f"[SETUP] Using device: {device}")
    return device


def create_objective(args) -> Any:
    """Create the IsaacLab objective function."""
    print(f"[SETUP] Creating {args.task} objective function...")
    
    objective = create_isaaclab_objective(
        task=args.task,
        train_timesteps=args.train_timesteps,
        eval_steps=args.eval_steps,
        eval_num_envs=args.eval_num_envs,
        n_eval_seeds=args.n_eval_seeds,
        verbose=args.verbose,
        isaaclab_path=args.isaaclab_path
    )
    
    print(f"[SETUP] Objective created with {objective.get_feature_count()} features")
    return objective


def test_objective_function(args) -> None:
    """Test the objective function and exit."""
    print(f"[TEST] Testing {args.task} objective function...")
    
    from .isaaclab_objective import test_isaaclab_objective
    
    # Test with reduced parameters
    results = test_isaaclab_objective(
        task=args.task,
        n_random_points=2,
        verbose=True
    )
    
    print("[TEST] Objective function test completed successfully!")
    print(f"[TEST] Tested {len(results['test_results'])} parameter configurations")
    
    if results['test_results'].numel() > 0:
        test_distances = results['test_results'].squeeze().tolist()
        if isinstance(test_distances, float):
            test_distances = [test_distances]
        print(f"[TEST] Distance results: {[f'{d:.1f}' for d in test_distances]}")


def run_optimization(args) -> Dict[str, Any]:
    """Run the main Bayesian optimization loop."""
    print(f"[BO] Starting Bayesian Optimization for {args.task}")
    print("=" * 60)
    
    # Create objective function
    objective = create_objective(args)
    
    # Setup device
    device = setup_device(args.device)
    
    # Create Bayesian optimizer
    print(f"[BO] Initializing optimizer...")
    optimizer = BayesianOptimizer(
        objective=objective,
        n_initial_points=args.n_initial_points,
        acquisition_function=args.acquisition,
        exploration_factor=args.exploration_factor,
        device=device
    )
    
    print(f"[BO] Configuration:")
    print(f"[BO]   Task: {args.task}")
    print(f"[BO]   Initial points: {args.n_initial_points}")
    print(f"[BO]   BO iterations: {args.n_iterations}")
    print(f"[BO]   Total evaluations: {args.n_initial_points + args.n_iterations}")
    print(f"[BO]   Training timesteps: {args.train_timesteps:,}")
    print(f"[BO]   Evaluation seeds: {args.n_eval_seeds}")
    print(f"[BO]   Acquisition: {args.acquisition}")
    print(f"[BO]   Device: {device}")
    print("=" * 60)
    
    # Run optimization
    start_time = time.time()
    results = optimizer.run_optimization(n_iterations=args.n_iterations)
    optimization_time = time.time() - start_time
    
    # Add timing and configuration info to results
    results.update({
        'task': args.task,
        'optimization_time': optimization_time,
        'total_evaluations': args.n_initial_points + args.n_iterations,
        'configuration': {
            'n_initial_points': args.n_initial_points,
            'n_iterations': args.n_iterations,
            'train_timesteps': args.train_timesteps,
            'eval_steps': args.eval_steps,
            'eval_num_envs': args.eval_num_envs,
            'n_eval_seeds': args.n_eval_seeds,
            'acquisition': args.acquisition,
            'exploration_factor': args.exploration_factor,
        }
    })
    
    return results


def save_and_analyze_results(results: Dict[str, Any], args) -> None:
    """Save results and generate analysis plots."""
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = args.task
    
    # Save main results
    results_file = results_dir / f"bo_results_{task_name}_{timestamp}.json"
    save_results(results, str(results_file))
    print(f"[SAVE] Results saved to: {results_file}")
    
    # Generate convergence plot
    try:
        convergence_file = results_dir / f"convergence_{task_name}_{timestamp}.png"
        plot_convergence(
            results['convergence_data'],
            title=f"BO Convergence - IsaacLab {task_name.title()}",
            save_path=str(convergence_file)
        )
        print(f"[SAVE] Convergence plot saved to: {convergence_file}")
    except Exception as e:
        print(f"[WARNING] Failed to generate convergence plot: {e}")
    
    # Generate feature importance plot
    try:
        if 'best_features' in results and results['best_features']:
            importance_file = results_dir / f"feature_importance_{task_name}_{timestamp}.png"
            plot_feature_importance(
                results['best_features'],
                title=f"Feature Importance - IsaacLab {task_name.title()}",
                save_path=str(importance_file)
            )
            print(f"[SAVE] Feature importance plot saved to: {importance_file}")
    except Exception as e:
        print(f"[WARNING] Failed to generate feature importance plot: {e}")


def print_summary(results: Dict[str, Any]) -> None:
    """Print optimization summary."""
    print("\n" + "=" * 70)
    print("🎉 BAYESIAN OPTIMIZATION COMPLETED!")
    print("=" * 70)
    
    print(f"📊 Task: {results['task'].title()}")
    print(f"🎯 Best distance: {results['best_value']:.3f}")
    print(f"⏱️  Optimization time: {results['optimization_time']:.1f}s ({results['optimization_time']/60:.1f} min)")
    print(f"🔍 Total evaluations: {results['total_evaluations']}")
    
    if 'best_features' in results and results['best_features']:
        print(f"\n🏆 Top 5 Most Important Features:")
        # Sort features by absolute weight
        sorted_features = sorted(
            results['best_features'].items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:5]
        
        for name, weight in sorted_features:
            print(f"   {name}: {weight:+.4f}")
    
    print(f"\n📈 Performance Summary:")
    if 'convergence_data' in results:
        conv_data = results['convergence_data']
        if 'best_values' in conv_data and len(conv_data['best_values']) > 0:
            initial_best = conv_data['best_values'][0]
            final_best = conv_data['best_values'][-1]
            improvement = final_best - initial_best
            print(f"   Initial best: {initial_best:.3f}")
            print(f"   Final best: {final_best:.3f}")
            print(f"   Improvement: {improvement:+.3f}")
    
    print("=" * 70)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Apply quick test overrides
    if args.quick_test:
        print("[QUICK-TEST] Applying reduced parameters for testing...")
        args.n_iterations = 3
        args.n_initial_points = 2
        args.train_timesteps = 50000  # ~25 iterations
        args.eval_steps = 1000
        args.n_eval_seeds = 1
        args.verbose = True
    
    try:
        # Test objective function if requested
        if args.test_objective:
            test_objective_function(args)
            return 0
        
        # Run optimization
        results = run_optimization(args)
        
        # Save and analyze results
        save_and_analyze_results(results, args)
        
        # Print summary
        print_summary(results)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Optimization interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Optimization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

