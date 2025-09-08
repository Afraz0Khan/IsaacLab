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
                 x_velocity: Optional[float] = None,
                 contact_forces: Optional[torch.Tensor] = None) -> float:
        # Simplified Gymnasium Ant reward
        healthy_reward = 1.0
        vx = 0.0 if x_velocity is None else float(x_velocity)
        forward_reward = 1.0 * vx
        ctrl_cost = 0.5 * float(np.sum(np.square(action)))
        contact_cost = 0.0
        if contact_forces is not None:
            f = np.asarray(contact_forces, dtype=np.float32).ravel()
            contact_cost = 5e-4 * float(np.sum(f * f))
        return float(healthy_reward + forward_reward - ctrl_cost - contact_cost)


class NegativeGroundAntReward(RewardFunc):
    """Negative of the proxy Ant ground-truth reward."""

    def __init__(self):
        self.gt = GroundTruthAntReward()

    def __call__(self,
                 env,
                 state: Optional[torch.Tensor],
                 action,
                 next_state,
                 x_velocity: Optional[float] = None,
                 contact_forces: Optional[torch.Tensor] = None) -> float:
        return -self.gt(env, state, action, next_state, x_velocity, contact_forces)

