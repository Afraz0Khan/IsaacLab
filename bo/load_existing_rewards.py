#!/usr/bin/env python3
"""
Load existing STARC seed reward functions for BO initialization.

This module loads the 20 seed reward functions from STARC results and converts
them to canonical format for use as the initial population in Bayesian optimization.
"""

import json
import os
import numpy as np
import torch
import importlib.util
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt


def _get_starc_env_name() -> str:
    # Allow selecting env via env var set by CLI (see --starc-env)
    return os.getenv("STARCV2_ENV", "halfcheetah")


def load_seed_reward_paths() -> List[str]:
    """Load the paths of seed reward functions from STARC results."""
    # Look for the seed rewards file
    current_dir = Path(__file__).parent
    env_name = _get_starc_env_name()
    possible_paths = [
        current_dir / "../starc_v2/results/T1/ga_seed_rewards.json",
        current_dir / "../../starc_v2/results/T1/ga_seed_rewards.json",
        Path("starc_v2/results/T1/ga_seed_rewards.json"),
        Path("../starc_v2/results/T1/ga_seed_rewards.json"),
        # Env-namespaced (new layout)
        current_dir / f"../starc_v2/results/{env_name}/T1/ga_seed_rewards.json",
        current_dir / f"../../starc_v2/results/{env_name}/T1/ga_seed_rewards.json",
        Path(f"starc_v2/results/{env_name}/T1/ga_seed_rewards.json"),
        Path(f"../starc_v2/results/{env_name}/T1/ga_seed_rewards.json"),
    ]
    
    for path in possible_paths:
        if path.exists():
            with open(path, 'r') as f:
                reward_paths = json.load(f)
            # Less verbose: only print count, not full path
            return reward_paths
    
    raise FileNotFoundError("Could not find ga_seed_rewards.json")


def load_canonical_features() -> Dict[str, str]:
    """Load the canonical feature definitions (seed_features preferred; fallback to features_simple).

    Resolution order:
      1) Environment variable CANONICAL_FEATURES_FILE
      2) starc_v2/scripts/seed_features.json
      3) starc_v2/scripts/features_simple.json
    """
    # 1) Env var override
    env_path = os.getenv("CANONICAL_FEATURES_FILE")
    if env_path and Path(env_path).exists():
        with open(env_path, 'r') as f:
            return json.load(f)

    current_dir = Path(__file__).parent
    # 2) Preferred seed_features.json
    seed_candidates = [
        current_dir / "../starc_v2/scripts/seed_features.json",
        current_dir / "../../starc_v2/scripts/seed_features.json",
        Path("starc_v2/scripts/seed_features.json"),
        Path("../starc_v2/scripts/seed_features.json"),
    ]
    for path in seed_candidates:
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)

    # 3) Fallback features_simple.json
    simple_candidates = [
        current_dir / "../starc_v2/scripts/features_simple.json",
        current_dir / "../../starc_v2/scripts/features_simple.json",
        Path("starc_v2/scripts/features_simple.json"),
        Path("../starc_v2/scripts/features_simple.json"),
    ]
    for path in simple_candidates:
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)

    raise FileNotFoundError("Could not find seed_features.json or features_simple.json")


def load_reward_function_from_file(reward_path: str):
    """Load a single reward function from its Python file."""
    # Find the actual file path
    current_dir = Path(__file__).parent
    possible_base_dirs = [
        current_dir / "..",  # From bo/ to openai/
        current_dir / "../..",  # From bo/ to parent of openai/
        Path("."),  # Current working directory 
        Path(".."),  # Parent of current working directory
    ]
    
    reward_file = None
    for base_dir in possible_base_dirs:
        # Try to find starc_v2 directory
        starc_dir = base_dir / "starc_v2"
        if starc_dir.exists():
            # Try to find the reward file under starc_v2
            full_path = starc_dir / reward_path
            if full_path.exists():
                reward_file = full_path
                break
    
    if reward_file is None:
        # Fallback: try direct path resolution
        for base_dir in possible_base_dirs:
            full_path = base_dir / reward_path
            if full_path.exists():
                reward_file = full_path
                break
    
    if reward_file is None:
        raise FileNotFoundError(f"Could not find reward file: {reward_path}")
    
    # Load the module
    spec = importlib.util.spec_from_file_location(
        f"reward_{reward_file.stem}", reward_file
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {reward_file}")
    
    module = importlib.util.module_from_spec(spec)
    
    # Add the starc_v2 directory to sys.path temporarily for imports
    starc_dir = reward_file.parent
    while starc_dir.name != "starc_v2" and starc_dir.parent != starc_dir:
        starc_dir = starc_dir.parent
    
    if starc_dir.name == "starc_v2" and str(starc_dir) not in sys.path:
        sys.path.insert(0, str(starc_dir))
    
    try:
        # Ensure LLM-generated rewards that subclass RewardFunc without import work
        try:
            from starc.core.reward_func import RewardFunc as _BaseRewardFunc  # type: ignore
            setattr(module, "RewardFunc", _BaseRewardFunc)
        except Exception:
            pass
        spec.loader.exec_module(module)
        
        # Find the reward class (should inherit from RewardFunc)
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and 
                hasattr(obj, '__bases__') and 
                any('RewardFunc' in str(base) for base in obj.__bases__)):
                return obj()
        
        raise ImportError(f"No RewardFunc class found in {reward_file}")
    
    finally:
        # Clean up sys.path
        if starc_dir.name == "starc_v2" and str(starc_dir) in sys.path:
            sys.path.remove(str(starc_dir))


def extract_weights_from_reward(reward_func) -> np.ndarray:
    """Extract weights from a STARC reward function."""
    try:
        # Try to get weights directly
        if hasattr(reward_func, 'weights'):
            return np.array(reward_func.weights)
        
        # Try to get from features and coefficients
        if hasattr(reward_func, 'get_features'):
            features = reward_func.get_features()
            if hasattr(reward_func, 'get_coefficients'):
                coeffs = reward_func.get_coefficients()
                return np.array(coeffs)
            elif hasattr(reward_func, 'coefficients'):
                return np.array(reward_func.coefficients)
        
        # Look for other weight attributes
        for attr in ['coeff', 'c', 'w', 'weight_vector']:
            if hasattr(reward_func, attr):
                return np.array(getattr(reward_func, attr))
        
        print(f"[SEED] Warning: Could not extract weights from {type(reward_func)}")
        print(f"[SEED] Available attributes: {[attr for attr in dir(reward_func) if not attr.startswith('_')]}")
        return np.array([])
        
    except Exception as e:
        print(f"[SEED] Warning: Error extracting weights: {e}")
        return np.array([])


def map_reward_to_canonical_weights(reward_func, canonical_features: Dict[str, str]) -> np.ndarray:
    """
    Map a STARC reward function to canonical feature weights.
    
    Args:
        reward_func: Loaded STARC reward function
        canonical_features: Dict of canonical feature definitions
        
    Returns:
        Array of weights for all canonical features (zero for unused features)
    """
    # Extract weights from the reward function
    reward_weights = extract_weights_from_reward(reward_func)
    
    if len(reward_weights) == 0:
        print(f"[SEED] Warning: No weights found, using zeros")
        return np.zeros(len(canonical_features))
    
    # Get the reward's features
    try:
        reward_features = reward_func.get_features() if hasattr(reward_func, 'get_features') else {}
    except Exception as e:
        print(f"[SEED] Warning: Could not get features: {e}")
        reward_features = {}
    
    # Create canonical weight vector (all zeros initially)
    canonical_weights = np.zeros(len(canonical_features))
    canonical_feature_names = list(canonical_features.keys())
    
    # Always map by feature names/expressions to keep sparsity
    for i, (feature_name, feature_expr) in enumerate(reward_features.items()):
        if i >= len(reward_weights):
            break
        # Find matching canonical feature
        canonical_idx = None
        for j, (canon_name, canon_expr) in enumerate(canonical_features.items()):
            if (feature_name == canon_name or 
                feature_expr == canon_expr or
                feature_name in canon_name or
                canon_name in feature_name):
                canonical_idx = j
                break
        if canonical_idx is not None:
            canonical_weights[canonical_idx] = reward_weights[i]
        else:
            print(f"[SEED] Warning: Could not map feature '{feature_name}' to canonical set")
    
    return canonical_weights


def load_seed_rewards_as_initial_points(max_rewards: Optional[int] = None, bound_expansion: float = 2.0) -> Tuple[List[np.ndarray], List[str], torch.Tensor]:
    """
    Load STARC seed reward functions and convert to canonical format.
    
    Args:
        max_rewards: Maximum number of rewards to load (None for all)
        bound_expansion: How much to expand bounds beyond STARC ranges (default: 2.0)
        
    Returns:
        Tuple of (initial_points, reward_names, dynamic_bounds)
        - initial_points: List of canonical weight arrays (original weights preserved)
        - reward_names: List of reward function names  
        - dynamic_bounds: Tensor of bounds [min(orig_weights) - expansion, max(orig_weights) + expansion] per feature
    """
    print(f"[SEED] Loading STARC seed reward functions...")
    
    # Load seed reward paths
    reward_paths = load_seed_reward_paths()
    
    if max_rewards is not None:
        reward_paths = reward_paths[:max_rewards]
    
    # Load canonical features
    canonical_features = load_canonical_features()
    n_features = len(canonical_features)
    
    initial_points = []
    reward_names = []
    loaded_count = 0
    failed_count = 0
    
    for i, reward_path in enumerate(reward_paths):
        try:
            # Load the reward function (less verbose)
            reward_func = load_reward_function_from_file(reward_path)
            
            # Convert to canonical weights
            canonical_weights = map_reward_to_canonical_weights(reward_func, canonical_features)

            # Keep original weights (no normalization) for meaningful scales
            if i == 0:
                max_abs = np.max(np.abs(canonical_weights))
                print(f"[SEED] Keeping original weights for {Path(reward_path).stem}: max_abs={max_abs:.3f}")

            initial_points.append(canonical_weights)
            reward_names.append(Path(reward_path).stem)
            loaded_count += 1
            
        except Exception as e:
            failed_count += 1
            # Only show failures, not individual successes
            if failed_count <= 3:  # Show first few failures
                print(f"[SEED] Failed to load {Path(reward_path).stem}: {e}")
            elif failed_count == 4:
                print(f"[SEED] ... (suppressing further failure messages)")
    
    if not initial_points:
        raise RuntimeError("No seed reward functions could be loaded successfully")
    
    # Convert to numpy array for bound calculation
    all_weights = np.array(initial_points)  # Shape: (n_rewards, n_features)
    
    # Compute dynamic bounds: [min(original_weights) - expansion, max(original_weights) + expansion] per feature
    min_bounds = np.min(all_weights, axis=0) - bound_expansion  # Min across all seed rewards, then -expansion
    max_bounds = np.max(all_weights, axis=0) + bound_expansion  # Max across all seed rewards, then +expansion
    
    # Convert to torch tensor for BO (use numpy array first for efficiency)
    bounds_array = np.array([min_bounds, max_bounds])
    dynamic_bounds = torch.from_numpy(bounds_array).to(dtype=torch.float64)
    
    print(f"[SEED] ✓ Loaded {loaded_count} seed rewards ({failed_count} failed)")
    print(f"[SEED] ✓ Dynamic bounds computed from original weights ±{bound_expansion}")
    print(f"[SEED] Dynamic BO bounds computed: ±{bound_expansion} from original ranges")
    print(f"[SEED] Feature bounds range from [{min_bounds.min():.3f}, {min_bounds.max():.3f}] to [{max_bounds.min():.3f}, {max_bounds.max():.3f}]")
    print(f"[SEED] Average active features per reward: {np.mean([np.sum(np.abs(w) > 1e-6) for w in all_weights]):.1f}")
    
    return initial_points, reward_names, dynamic_bounds


def plot_seed_rewards(initial_points: List[np.ndarray], 
                     reward_names: List[str], 
                     canonical_features: Dict[str, str],
                     save_path: Optional[str] = None):
    """
    Plot all seed reward functions showing their feature weights.
    
    Args:
        initial_points: List of weight vectors
        reward_names: List of reward function names
        canonical_features: Dict of feature definitions
        save_path: Optional path to save the plot
    """
    n_rewards = len(initial_points)
    n_features = len(canonical_features)
    feature_names = list(canonical_features.keys())
    
    # Create figure
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))  # 4x5 grid for 20 rewards
    axes = axes.flatten()
    
    for i, (weights, name) in enumerate(zip(initial_points, reward_names)):
        ax = axes[i]
        
        # Get active features (non-zero weights)
        active_indices = np.where(np.abs(weights) > 1e-6)[0]
        active_features = [feature_names[j] for j in active_indices]
        active_weights = weights[active_indices]
        
        if len(active_features) > 0:
            # Create bar plot
            colors = ['green' if w > 0 else 'red' for w in active_weights]
            bars = ax.bar(range(len(active_features)), active_weights, color=colors, alpha=0.7)
            
            # Set labels
            ax.set_title(f"{name}\n{len(active_features)} active features", fontsize=10)
            ax.set_xticks(range(len(active_features)))
            ax.set_xticklabels(active_features, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Weight', fontsize=8)
            ax.set_ylim([-1.1, 1.1])
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, weight in zip(bars, active_weights):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height > 0 else -0.1),
                       f'{weight:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=7)
        else:
            ax.text(0.5, 0.5, 'No active features', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{name}\n0 active features", fontsize=10)
    
    # Hide empty subplots
    for i in range(n_rewards, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SEED] Seed reward plot saved to: {save_path}")
    else:
        plt.show()
    
    return fig


def get_seed_reward_info() -> Dict:
    """Get information about available seed reward functions."""
    try:
        reward_paths = load_seed_reward_paths()
        canonical_features = load_canonical_features()
        
        return {
            "available": True,
            "count": len(reward_paths),
            "paths": reward_paths,
            "n_canonical_features": len(canonical_features),
            "canonical_feature_names": list(canonical_features.keys())
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
            "count": 0
        }


if __name__ == "__main__":
    # Demo: Load and plot all seed rewards
    print("Loading STARC seed reward functions...")
    
    try:
        initial_points, reward_names, dynamic_bounds = load_seed_rewards_as_initial_points()
        canonical_features = load_canonical_features()
        
        print(f"\nSuccessfully loaded {len(initial_points)} seed rewards")
        print(f"Canonical features: {len(canonical_features)}")
        
        # Show dynamic bounds info
        bounds_cpu = dynamic_bounds.cpu().numpy()
        print(f"Dynamic bounds computed: [{bounds_cpu[0].min():.3f}, {bounds_cpu[0].max():.3f}] to [{bounds_cpu[1].min():.3f}, {bounds_cpu[1].max():.3f}]")
        
        # Plot them
        plot_seed_rewards(
            initial_points, 
            reward_names, 
            canonical_features,
            save_path="seed_rewards_plot.png"
        )
        
        # Show summary statistics
        all_weights = np.array(initial_points)
        active_counts = [np.sum(np.abs(weights) > 1e-6) for weights in initial_points]
        
        print(f"\nSummary statistics:")
        print(f"Average active features: {np.mean(active_counts):.1f} ± {np.std(active_counts):.1f}")
        print(f"Weight range: [{all_weights.min():.3f}, {all_weights.max():.3f}]")
        print(f"Average weight magnitude: {np.mean(np.abs(all_weights[all_weights != 0])):.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 