from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_42(RewardFunc):
    """
    Emphasises forward acceleration squared.
    features: [x_vel, accel_sq, ctrl_cost]
    """
    def __init__(self):
        self._w = np.array([1.0, 0.6, -0.05])

    def __call__(self, env, state, action, next_state, x_velocity):
        if x_velocity is None: x_velocity=state[9]
        next_vx = next_state[9] if next_state is not None else x_velocity
        accel_sq = (next_vx - state[9])**2
        ctrl_cost = np.sum(np.square(action))
        r = self._w[0]*x_velocity + self._w[1]*accel_sq + self._w[2]*ctrl_cost
        return r.item() if hasattr(r,"item") else r

    @property
    def weights(self): return self._w
    def set_weights(self, w): self._w = w
    def get_features(self):
        return {
            "x_vel":"x_velocity",
            "accel_sq":"(next_state[9]-state[9])**2",
            "ctrl_cost":"np.sum(np.square(action))"
        }