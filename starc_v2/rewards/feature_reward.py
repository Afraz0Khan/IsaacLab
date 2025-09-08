"""
Feature-based reward base class for STARC v2 and IsaacLab locomotion experiments.

Usage for LLM-generated rewards:
- Subclass FeatureRewardFunc
- Implement compute_features(env, state, action, next_state) → Dict[str, float]
- Optionally override default_weights() to expose canonical feature names

Compatibility:
- Inherits from starc.core.reward_func.RewardFunc, so it plugs into STARC analyzers.
- __call__ returns a float reward; additional kwargs may be ignored by callers.

Notes on inputs:
- env: placeholder for environment context (may be None in offline analysis)
- state / next_state: frameworks may pass dicts, numpy arrays, or tensors
- action: numpy array / list / tensor for that transition
"""

from __future__ import annotations

from typing import Dict, Any
import abc
import math

from starc.core.reward_func import RewardFunc


class FeatureRewardFunc(RewardFunc):
    """
    Base class that turns a dictionary of feature values into a scalar reward via weights.

    Implementors should:
    - override compute_features to return a dictionary mapping feature names to floats
    - optionally override default_weights to advertise canonical features for this class

    Canonical locomotion feature names (recommended):
    - forward_velocity: +X linear velocity in world frame
    - speed: L2 norm of linear velocity
    - upright: dot(body_up, world_up) or yaw-only heading alignment
    - move_to_target: cosine heading or progress toward a target point
    - action_l2: sum(|action|) or L2 norm of torques/commands (typically penalized)
    - energy: sum(|tau * dq|) or approximate energy usage (penalized)
    - smoothness: action or velocity change penalty between steps (penalized)
    - joint_pos_limits: penalty near joint limits
    - torso_height: torso/root height (often for stability)
    """

    def __init__(self, weights: Dict[str, float] | None = None):
        self.weights: Dict[str, float] = weights.copy() if isinstance(weights, dict) else {}

    # --- Interface to implement -------------------------------------------------
    @abc.abstractmethod
    def compute_features(self, env: Any, state: Any, action: Any, next_state: Any) -> Dict[str, float]:
        """Return a dictionary of feature_name → float value for this transition.

        Implementations should not mutate inputs and should handle numpy/tensor inputs gracefully.
        """

    # --- Optional helpers -------------------------------------------------------
    def default_weights(self) -> Dict[str, float]:
        """Advertise canonical features and default weights (may be overridden)."""
        return {}

    # --- Composition to scalar reward ------------------------------------------
    def __call__(self, env, state, action, next_state) -> float:  # type: ignore[override]
        feats = self.compute_features(env, state, action, next_state)
        # Combine with weights (missing weights default to 0.0)
        reward = 0.0
        for name, value in feats.items():
            w = float(self.weights.get(name, 0.0))
            try:
                v = float(value)
            except Exception:
                # Best effort: handle scalar tensors/arrays
                try:
                    v = float(getattr(value, "item", lambda: value)())
                except Exception:
                    v = 0.0
            reward += w * v
        # Guard against NaNs
        if math.isnan(reward) or math.isinf(reward):
            return 0.0
        return float(reward)


