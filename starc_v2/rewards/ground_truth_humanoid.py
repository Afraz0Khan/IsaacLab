from typing import Optional
import numpy as np
import torch
import sys
import os

# Allow import of base RewardFunc from original starc
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'starc'))
from starc.core.reward_func import RewardFunc  # type: ignore


class GroundTruthHumanoidReward(RewardFunc):
    """
    Simplified proxy ground-truth for Humanoid locomotion used as STARC reference:
    reward = forward_velocity - 0.001 * sum(|action|) - 2.0 * fall_penalty

    fall_penalty is approximated as a large penalty if next_state has NaNs or
    if vertical body component appears invalid. This remains environment-agnostic.
    """

    def __call__(self,
                 env,
                 state: Optional[torch.Tensor],
                 action,
                 next_state,
                 x_velocity: float = None) -> float:
        if x_velocity is None:
            x_velocity = 0.0
        try:
            act = np.asarray(action, dtype=np.float32).reshape(-1)
            energy_penalty = 0.001 * float(np.sum(np.abs(act)))
        except Exception:
            energy_penalty = 0.0
        fall_term = 0.0
        try:
            ns = np.asarray(next_state, dtype=np.float32).reshape(-1)
            if not np.all(np.isfinite(ns)):
                fall_term = 1.0
        except Exception:
            pass
        return float(x_velocity) - energy_penalty - 2.0 * fall_term


class NegativeGroundHumanoidReward(RewardFunc):
    """Negative of the proxy Humanoid ground-truth reward."""

    def __init__(self):
            self.gt = GroundTruthHumanoidReward()

    def __call__(self,
                 env,
                 state: Optional[torch.Tensor],
                 action,
                 next_state,
                 x_velocity: float = None) -> float:
        return -self.gt(env, state, action, next_state, x_velocity)

