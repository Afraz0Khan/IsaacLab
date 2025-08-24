from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_44(RewardFunc):
    """
    Uses cosine-weighted speed for upright running.
    features: [cos_speed, ctrl_cost]
    """
    def __init__(self):
        self._w = np.array([1.3, -0.05])

    def __call__(self, env, state, action, next_state, x_velocity):
        if x_velocity is None: x_velocity=state[9]
        cos_speed = x_velocity * np.cos(state[2])
        ctrl_cost = np.sum(np.square(action))
        r = self._w[0]*cos_speed + self._w[1]*ctrl_cost
        return r.item() if hasattr(r,"item") else r

    @property
    def weights(self): return self._w
    def set_weights(self,w): self._w=w
    def get_features(self):
        return {
            "cos_speed":"x_velocity*np.cos(state[2])",
            "ctrl_cost":"np.sum(np.square(action))"
        }