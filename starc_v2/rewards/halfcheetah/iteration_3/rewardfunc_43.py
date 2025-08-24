from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_43(RewardFunc):
    """
    Speed weighted by foot angle stability.
    features: [speed_mod, ctrl_cost]
    """
    def __init__(self):
        self._w = np.array([1.0, -0.05])

    def __call__(self, env, state, action, next_state, x_velocity):
        if x_velocity is None: x_velocity=state[9]
        abs_front_foot_angle = np.abs(state[8])
        speed_mod = x_velocity / (1.0 + abs_front_foot_angle)
        ctrl_cost = np.sum(np.square(action))
        r = self._w[0]*speed_mod + self._w[1]*ctrl_cost
        return r.item() if hasattr(r,"item") else r

    @property
    def weights(self): return self._w
    def set_weights(self,w): self._w = w
    def get_features(self):
        return {
            "speed_mod":"x_velocity/(1.0+np.abs(state[8]))",
            "ctrl_cost":"np.sum(np.square(action))"
        }