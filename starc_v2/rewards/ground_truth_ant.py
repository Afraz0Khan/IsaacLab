from typing import Optional
import numpy as np
import torch
import sys
import os

# Allow import of base RewardFunc from original starc
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'starc'))
from starc.core.reward_func import RewardFunc  # type: ignore


class GroundTruthAntReward(RewardFunc):
    """
    Simplified proxy ground-truth for Ant locomotion used as STARC reference:
    reward = forward_velocity - 0.001 * sum(|action|)

    Notes:
    - Does not rely on env-specific internals; uses provided x_velocity and action.
    - Signature matches RewardFunc for compatibility with STARC analyzer.
    """

    def __call__(self,
                 env,  # unused placeholder for signature compatibility
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
        return float(x_velocity) - energy_penalty


class NegativeGroundAntReward(RewardFunc):
    """Negative of the proxy Ant ground-truth reward."""

    def __init__(self):
        self.gt = GroundTruthAntReward()

    def __call__(self,
                 env,
                 state: Optional[torch.Tensor],
                 action,
                 next_state,
                 x_velocity: float = None) -> float:
        return -self.gt(env, state, action, next_state, x_velocity)

