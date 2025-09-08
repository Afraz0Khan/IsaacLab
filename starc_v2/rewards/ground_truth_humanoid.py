from typing import Optional, cast
import numpy as np
import torch
import sys
import os

# Allow import of base RewardFunc from original starc
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'starc'))
from starc.core.reward_func import RewardFunc  # type: ignore


class GroundTruthHumanoidReward(RewardFunc):
    """Gymnasium-style Humanoid reward proxy for STARC analysis.

    reward = healthy_reward + forward_reward - ctrl_cost - contact_cost

    Defaults (Gymnasium Humanoid):
    - healthy_reward = 5.0 per step
    - forward_reward_weight = 1.25
    - ctrl_cost_weight = 0.1
    - contact_cost_weight = 5e-7

    Contact forces are not present in STARC transitions, so contact_cost=0 unless
    caller provides next_state["contact_forces"].
    """

    def __call__(self,
                 env,
                 state: Optional[torch.Tensor],
                 action,
                 next_state,
                 x_velocity: Optional[float] = None,
                 contact_forces: Optional[torch.Tensor] = None) -> float:
        # Simplified Gymnasium Humanoid reward
        healthy_reward = 5.0
        vx = cast(float, x_velocity)
        forward_reward = 1.25 * vx
        ctrl_cost = 0.1 * float(np.sum(np.square(action)))
        contact_cost = 0.0
        if contact_forces is not None:
            f = np.asarray(contact_forces, dtype=np.float32).ravel()
            contact_cost = 5e-7 * float(np.sum(f * f))
        return float(healthy_reward + forward_reward - ctrl_cost - contact_cost)


class NegativeGroundHumanoidReward(RewardFunc):
    """Negative of the proxy Humanoid ground-truth reward."""

    def __init__(self):
            self.gt = GroundTruthHumanoidReward()

    def __call__(self,
                 env,
                 state: Optional[torch.Tensor],
                 action,
                 next_state,
                 x_velocity: Optional[float] = None,
                 contact_forces: Optional[torch.Tensor] = None) -> float:
        return -self.gt(env, state, action, next_state, x_velocity, contact_forces)

