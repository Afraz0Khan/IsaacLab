from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_11(RewardFunc):
    """
    Adds a quadratic speed term to favour very high speeds.
    features: [x_velocity, x_vel_sq, ctrl_cost]
    """
    def __init__(self):
        self._w = np.array([1.1, 0.20, -0.05])

    def __call__(self, env, state, action, next_state, x_velocity):
        if x_velocity is None: x_velocity = state[9]
        x_vel_sq = x_velocity**2
        ctrl_cost = np.sum(np.square(action))
        r = self._w[0]*x_velocity + self._w[1]*x_vel_sq + self._w[2]*ctrl_cost
        if hasattr(r,'item'): return r.item()
        return r

    @property
    def weights(self): return self._w
    def set_weights(self,w): self._w=w
    def get_features(self):
        return {
            "x_velocity":"x_velocity",
            "x_vel_sq":"x_velocity**2",
            "ctrl_cost":"np.sum(np.square(action))"
        }