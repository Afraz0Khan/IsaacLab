#!/usr/bin/env python3
"""
Canonical reward function for Bayesian optimization.

This module defines a canonical sum-of-features reward function that follows
the standard STARC reward function pattern. Features are loaded from 
seed_features.json and combined with learnable weights in the range [-1, 1].
"""

import json
import os
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union, Dict


class CanonicalReward:
    """
    A canonical reward function that follows the standard STARC pattern.
    
    The reward function is a linear combination of all canonical features:
    R(s, a, s') = Σ w_i * f_i(s, a, s')
    
    where f_i are canonical features and w_i are learnable weights in [-1, 1].
    """
    
    def __init__(self, weights: Union[np.ndarray, torch.Tensor], 
                 features_file: Optional[str] = None):
        """
        Initialize the canonical reward function.
        
        Args:
            weights: Array of weights for each feature (range [-1, 1])
            features_file: Path to seed_features.json file (auto-detected if None)
        """
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        
        self._weights = np.asarray(weights, dtype=np.float32)
        
        # Load features from JSON file
        self._features = self._load_features(features_file)
        self._feature_names = list(self._features.keys())
        # Pre-compile feature expressions for speed and set up reusable context
        self._feature_exprs = [self._features[name] for name in self._feature_names]
        self._compiled_exprs = [compile(expr, '<canonical_feature>', 'eval') for expr in self._feature_exprs]
        # Reusable eval context; update state/action/x_velocity per step
        self._eval_ctx = {
            'np': np,
            'abs': abs,
            'state': None,
            'action': None,
            'x_velocity': None,
        }
        
        # Validate weights dimension
        if len(self._weights) != len(self._feature_names):
            raise ValueError(f"Number of weights ({len(self._weights)}) must match "
                           f"number of features ({len(self._feature_names)})")
        
        # Clamp weights to [-1, 1] range
        self._weights = np.clip(self._weights, -1.0, 1.0)
        
    def _load_features(self, features_file: Optional[str] = None) -> Dict[str, str]:
        """Load canonical feature definitions (JSON mapping name -> expression).

        Resolution order:
        1) Explicit path argument
        2) Environment variable CANONICAL_FEATURES_FILE
        3) starc_v2/scripts/seed_features.json (preferred)
        4) starc_v2/scripts/features_simple.json (fallback)
        """
        # 1) explicit
        if features_file is not None:
            features_path = Path(features_file)
            if not features_path.exists():
                raise FileNotFoundError(f"Features file not found: {features_path}")
        else:
            # 2) env var (fallback to auto-detect if invalid)
            env_path = os.getenv("CANONICAL_FEATURES_FILE")
            if env_path:
                p = Path(env_path)
                if p.exists():
                    features_path = p
                else:
                    # Do not raise here; gracefully fall back to auto-detect below
                    features_path = None
            else:
                features_path = None
            # 3/4) auto-detect (preferred seed_features.json, then features_simple.json)
            if features_path is None:
                current_dir = Path(__file__).parent
                candidates = [
                    # preferred seed_features.json
                    current_dir / "../starc_v2/scripts/seed_features.json",
                    current_dir / "../../starc_v2/scripts/seed_features.json",
                    current_dir / "../../../starc_v2/scripts/seed_features.json",
                    Path("starc_v2/scripts/seed_features.json"),
                    Path("../starc_v2/scripts/seed_features.json"),
                    # fallback to features_simple.json
                    current_dir / "../starc_v2/scripts/features_simple.json",
                    current_dir / "../../starc_v2/scripts/features_simple.json",
                    current_dir / "../../../starc_v2/scripts/features_simple.json",
                    Path("starc_v2/scripts/features_simple.json"),
                    Path("../starc_v2/scripts/features_simple.json"),
                ]
                for c in candidates:
                    if c.exists():
                        features_path = c
                        break
                if features_path is None:
                    raise FileNotFoundError(
                        "Could not find canonical features JSON. Set CANONICAL_FEATURES_FILE or place "
                        "seed_features.json/features_simple.json under starc_v2/scripts/."
                    )

        with open(features_path, 'r') as f:
            features = json.load(f)
        return features
    
    def __call__(self, env, state: Optional[np.ndarray], action: np.ndarray, 
                 next_state: np.ndarray, x_velocity: Optional[float] = None) -> float:
        """
        Compute the reward given the current transition.
        
        Args:
            env: The HalfCheetah environment
            state: Current state (can be None for first step)
            action: Action taken
            next_state: Resulting next state
            x_velocity: Forward velocity (required for most features)
            
        Returns:
            Computed reward value as linear combination of weighted features
        """
        if state is None or x_velocity is None:
            return 0.0
            
        # Convert tensors to numpy if needed
        if hasattr(action, 'cpu'):
            action = action.cpu().numpy()  
        if hasattr(next_state, 'cpu'):
            next_state = next_state.cpu().numpy()
        
        # Update reusable eval context once per call
        ctx = self._eval_ctx
        ctx['state'] = next_state
        ctx['action'] = action
        ctx['x_velocity'] = x_velocity
        # Evaluate each feature and compute weighted sum using precompiled code
        reward = 0.0
        for i, code_obj in enumerate(self._compiled_exprs):
            wi = self._weights[i]
            if abs(wi) <= 1e-8:
                continue
            try:
                val = eval(code_obj, {"__builtins__": {}}, ctx)
                # Convert numpy scalar to float if needed
                if hasattr(val, 'item'):
                    val = val.item()
                if np.isfinite(val):
                    reward += wi * float(val)
            except Exception:
                # Skip invalid feature
                continue
        
        return float(reward)
    
    # _evaluate_feature no longer used; kept for compatibility
    def _evaluate_feature(self, feature_expr: str, state: np.ndarray, 
                         action: np.ndarray, next_state: np.ndarray, 
                         x_velocity: float) -> float:
        ctx = self._eval_ctx
        ctx['state'] = next_state
        ctx['action'] = action
        ctx['x_velocity'] = x_velocity
        try:
            val = eval(feature_expr, {"__builtins__": {}}, ctx)
            if hasattr(val, 'item'):
                val = val.item()
            return float(val) if np.isfinite(val) else 0.0
        except Exception:
            return 0.0
    
    @property
    def weights(self) -> np.ndarray:
        """Get current weights (following STARC standard)."""
        return self._weights.copy()
    
    def set_weights(self, weights: np.ndarray) -> None:
        """Set new weights (following STARC standard)."""
        if len(weights) != len(self._weights):
            raise ValueError(f"Wrong number of weights: expected {len(self._weights)}, got {len(weights)}")
        self._weights = np.clip(np.asarray(weights, dtype=np.float32), -1.0, 1.0)
    
    def get_features(self) -> Dict[str, str]:
        """Get feature definitions (following STARC standard)."""
        return self._features.copy()
    
    def get_feature_names(self) -> list:
        """Get list of feature names."""
        return self._feature_names.copy()
    
    @staticmethod
    def get_bounds():
        """
        Get parameter bounds for Bayesian optimization.
        
        Returns:
            bounds: torch.Tensor of shape (2, n_features) with [lower_bounds, upper_bounds]
        """
        # Load features to get the dimension
        try:
            # If env var provided, honor it
            env_path = os.getenv("CANONICAL_FEATURES_FILE")
            if env_path and Path(env_path).exists():
                with open(env_path, 'r') as f:
                    features = json.load(f)
                n_features = len(features)
            else:
                current_dir = Path(__file__).parent
                candidates = [
                    current_dir / "../starc_v2/scripts/seed_features.json",
                    current_dir / "../../starc_v2/scripts/seed_features.json",
                    current_dir / "../../../starc_v2/scripts/seed_features.json",
                    Path("starc_v2/scripts/seed_features.json"),
                    Path("../starc_v2/scripts/seed_features.json"),
                    current_dir / "../starc_v2/scripts/features_simple.json",
                    current_dir / "../../starc_v2/scripts/features_simple.json",
                    current_dir / "../../../starc_v2/scripts/features_simple.json",
                    Path("starc_v2/scripts/features_simple.json"),
                    Path("../starc_v2/scripts/features_simple.json"),
                ]
                for c in candidates:
                    if c.exists():
                        with open(c, 'r') as f:
                            features = json.load(f)
                        n_features = len(features)
                        break
                else:
                    # last resort
                    n_features = 21
        except Exception:
            n_features = 21
        
        # All weights are bounded in [-1, 1]
        lower_bounds = torch.full((n_features,), -1.0, dtype=torch.float64)
        upper_bounds = torch.full((n_features,), 1.0, dtype=torch.float64)
        
        bounds = torch.stack([lower_bounds, upper_bounds])
        return bounds
    
    @property
    def dim(self):
        """Return the dimension of the parameter space."""
        return len(self._feature_names)


# Alias for backward compatibility
ParameterizedReward = CanonicalReward 