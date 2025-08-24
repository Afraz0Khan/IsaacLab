from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_1(RewardFunc):
    """
    Simple re-weighting of the expert: more emphasis on running fast and a milder
    control penalty.
    features: [x_velocity, ctrl_cost]
    """
    def __init__(self):
        self._w = np.array([1.2, -0.05])

    def __call__(self,
                 env: HalfCheetahEnv,
                 state: Optional[torch.Tensor],
                 action,
                 next_state,
                 x_velocity: Optional[float]) -> float:
        if x_velocity is None:
            x_velocity = state[9]
        ctrl_cost = np.sum(np.square(action))
        reward = self._w[0]*x_velocity + self._w[1]*ctrl_cost
        if hasattr(reward, "item"): return reward.item()
        return reward

    @property
    def weights(self) -> np.ndarray:
        return self._w

    def set_weights(self, w: np.ndarray):
        self._w = w

    def get_features(self) -> Dict[str, str]:
        return {
            "x_velocity": "x_velocity",
            "ctrl_cost": "np.sum(np.square(action))"
        }