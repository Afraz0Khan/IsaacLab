#!/usr/bin/env python3
"""
Example script demonstrating HalfCheetah canonical reward function optimization.

This example shows how to use the BO framework to discover optimal reward functions
using random initialization with [-1, 1] bounds for all feature weights.
"""

import numpy as np
from pathlib import Path

from .reward_function import CanonicalReward
from .objective import HalfCheetahObjective
from .bayesian_optimization import BayesianOptimizer
from .run_optimization import run_halfcheetah_bo
from .train_evaluate import train_and_evaluate, test_baseline_performance


def demo_canonical_reward():
    """Demonstrate how to create and use canonical reward functions."""
    print("="*60)
    print("DEMO: CANONICAL REWARD FUNCTION")
    print("="*60)
    
    # Create canonical reward with random weights
    reward = CanonicalReward()
    
    print(f"Canonical reward has {len(reward.feature_names)} features")
    print(f"Feature names: {list(reward.feature_names.keys())[:5]}... (first 5)")
    
    # Set random weights in [-1, 1] range
    random_weights = np.random.uniform(-1, 1, len(reward.feature_names))
    reward.set_weights(random_weights)
    
    print(f"Set random weights with {np.sum(np.abs(random_weights) > 1e-6)} active features")
    print(f"Weight range: [{random_weights.min():.3f}, {random_weights.max():.3f}]")
    print(f"Weight magnitude: {np.linalg.norm(random_weights):.3f}")
    
    # Show top features by weight magnitude
    active_features = [(name, weight) for name, weight in zip(reward.feature_names.keys(), random_weights) 
                      if abs(weight) > 1e-6]
    active_features.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\nTop 5 features by weight magnitude:")
    for name, weight in active_features[:5]:
        print(f"  {name}: {weight:+.4f}")
    
    print("✓ Canonical reward demonstration complete\n")


def demo_objective_function():
    """Demonstrate objective function evaluation."""
    print("="*60)
    print("DEMO: OBJECTIVE FUNCTION EVALUATION")
    print("="*60)
    
    # Create objective with fast settings for demo
    objective = HalfCheetahObjective(
        train_timesteps=5000,  # Fast for demo
        eval_episodes=2,       # Minimal evaluation
        eval_timesteps=500,
        verbose=True
    )
    
    print(f"Objective function evaluates {objective.get_feature_count()} features")
    
    # Test with a few random weight vectors
    print("\nTesting with 2 random weight vectors:")
    
    for i in range(2):
        # Generate random weights in [-1, 1]
        weights = np.random.uniform(-1, 1, objective.get_feature_count())
        active_count = np.sum(np.abs(weights) > 1e-6)
        
        print(f"\n[TEST {i+1}] Evaluating weights with {active_count} active features...")
        print(f"Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
        
        # Evaluate the objective
        distance = objective.evaluate(weights)
        print(f"Result: Distance = {distance:.4f}")
    
    print("\n✓ Objective function demonstration complete\n")


def demo_bayesian_optimization():
    """Demonstrate a quick Bayesian optimization run."""
    print("="*60)
    print("DEMO: BAYESIAN OPTIMIZATION")
    print("="*60)
    
    print("Running quick BO demonstration (reduced parameters for speed)...")
    
    # Quick BO run with minimal parameters
    results = run_halfcheetah_bo(
        n_iterations=3,         # Very few iterations for demo
        n_initial_points=2,     # Few initial points
        train_timesteps=5000,   # Fast training
        eval_episodes=2,        # Quick evaluation
        eval_timesteps=500,
        acquisition_function='UCB',
        exploration_factor=1.0,
        test_baseline=True,
        results_dir="demo_results",
        verbose=True
    )
    
    print(f"\n✓ BO demonstration complete!")
    print(f"Best distance found: {results['best_value']:.4f}")
    print(f"Total evaluations: {results['total_evaluations']}")
    print(f"Active features in best solution: {len([w for w in results['best_weights'] if abs(w) > 1e-6])}")
    
    return results


def demo_feature_analysis():
    """Demonstrate feature analysis capabilities."""
    print("="*60)
    print("DEMO: FEATURE ANALYSIS")
    print("="*60)
    
    # Create objective for analysis
    objective = HalfCheetahObjective(
        train_timesteps=1000,   # Minimal for demo
        eval_episodes=1,
        eval_timesteps=100,
        verbose=False
    )
    
    # Generate several random weight vectors and analyze them
    n_samples = 3
    results = []
    
    print(f"Analyzing {n_samples} random weight configurations...")
    
    for i in range(n_samples):
        weights = np.random.uniform(-1, 1, objective.get_feature_count())
        distance = objective.evaluate(weights)
        
        active_features = objective.get_active_features(weights)
        
        results.append({
            'weights': weights,
            'distance': distance,
            'active_features': active_features,
            'n_active': len(active_features)
        })
        
        print(f"Sample {i+1}: Distance={distance:.3f}, Active features={len(active_features)}")
    
    # Find best result
    best_result = max(results, key=lambda x: x['distance'])
    
    print(f"\nBest configuration:")
    print(f"  Distance: {best_result['distance']:.4f}")
    print(f"  Active features: {best_result['n_active']}")
    print(f"  Top features:")
    for name, weight in list(best_result['active_features'].items())[:3]:
        print(f"    {name}: {weight:+.4f}")
    
    print("✓ Feature analysis demonstration complete\n")


def full_demo():
    """Run the complete demonstration."""
    print("🚀 HALFCHEETAH CANONICAL REWARD OPTIMIZATION DEMO")
    print("This demo shows the key components of the BO framework\n")
    
    try:
        # Run individual demos
        demo_canonical_reward()
        demo_objective_function() 
        demo_feature_analysis()
        
        # Skip the BO demo if we want to keep it really fast
        print("="*60)
        print("DEMO: BAYESIAN OPTIMIZATION (OPTIONAL)")
        print("="*60)
        
        response = input("Run BO demo? This will take a few minutes (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            demo_results = demo_bayesian_optimization()
        else:
            print("Skipping BO demonstration.")
        
        print("\n🎉 ALL DEMONSTRATIONS COMPLETE!")
        print("You can now run the full optimization with:")
        print("  python -m bo.run_optimization --n-iterations 50")
        print("  python -m bo.run_optimization --quick-test  # For faster testing")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def simple_example():
    """Simple usage example for documentation."""
    print("SIMPLE USAGE EXAMPLE")
    print("="*40)
    
    # Basic usage
    from bo import run_halfcheetah_bo
    
    print("Running optimization with default parameters...")
    
    # This would run the full optimization (commented out for demo)
    # results = run_halfcheetah_bo()
    
    print("# For actual usage, uncomment the line above")
    print("# This will run full BO with default parameters:")
    print("#   n_iterations=50, n_initial_points=10")
    print("#   train_timesteps=80000, eval_episodes=10")
    print("#   bounds=[-1, 1] for all feature weights")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "simple":
            simple_example()
        elif sys.argv[1] == "canonical":
            demo_canonical_reward()
        elif sys.argv[1] == "objective":
            demo_objective_function()
        elif sys.argv[1] == "features":
            demo_feature_analysis()
        elif sys.argv[1] == "bo":
            demo_bayesian_optimization()
        else:
            print("Usage: python example.py [simple|canonical|objective|features|bo]")
    else:
        full_demo() 