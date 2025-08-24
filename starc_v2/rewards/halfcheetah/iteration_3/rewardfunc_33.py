from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_33(RewardFunc):
    """
    Exponential speed incentive with quadratic torque cost.
    features: [exp_speed, ctrl_cost]
    """
    def __init__(self):
        self._w = np.array([0.8, -0.05])

    def __call__(self, env: HalfCheetahEnv, state, action, next_state, x_velocity):
        if x_velocity is None: x_velocity = state[9]
        exp_speed = np.exp(0.3 * x_velocity)
        ctrl_cost = np.sum(np.square(action))
        r = self._w[0] * exp_speed + self._w[1] * ctrl_cost
        return r.item() if hasattr(r, "item") else r

    @property
    def weights(self): return self._w
    def set_weights(self, w): self._w = w
    def get_features(self):
        return {
            "exp_speed": "np.exp(0.3 * x_velocity)",
            "ctrl_cost": "np.sum(np.square(action))"
        }