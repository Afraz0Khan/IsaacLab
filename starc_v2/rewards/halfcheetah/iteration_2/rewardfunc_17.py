from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_17(RewardFunc):
    """
    Speed, torque penalty, and efficiency (speed per torque).
    features: [x_vel, ctrl_cost, efficiency]
    """
    def __init__(self):
        self._w = np.array([1.0, -0.50, 0.20])

    def __call__(self, env: HalfCheetahEnv, state, action, next_state, x_velocity):
        if x_velocity is None:
            x_velocity = state[9]
        ctrl_cost = np.sum(np.square(action))
        efficiency = x_velocity / (ctrl_cost + 1e-6)
        r = self._w[0]*x_velocity + self._w[1]*ctrl_cost + self._w[2]*efficiency
        return r.item() if hasattr(r, "item") else r

    @property
    def weights(self): return self._w
    def set_weights(self, w): self._w = w
    def get_features(self):
        return {
            "x_vel": "x_velocity",
            "ctrl_cost": "np.sum(np.square(action))",
            "efficiency": "x_velocity/(ctrl_cost+1e-6)"
        }