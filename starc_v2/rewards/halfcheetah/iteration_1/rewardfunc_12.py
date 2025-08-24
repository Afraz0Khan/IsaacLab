from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_12(RewardFunc):
    """
    Uses tanh(speed) to reduce diminishing returns at very high speed,
    keeping incentives smooth.
    features: [tanh_v, ctrl_cost]
    """
    def __init__(self):
        self._w = np.array([2.0, -0.06])

    def __call__(self, env, state, action, next_state, x_velocity):
        if x_velocity is None: x_velocity = state[9]
        tanh_v = np.tanh(x_velocity)
        ctrl_cost = np.sum(np.square(action))
        r = self._w[0]*tanh_v + self._w[1]*ctrl_cost
        if hasattr(r,'item'): return r.item()
        return r

    @property
    def weights(self): return self._w
    def set_weights(self,w): self._w=w
    def get_features(self):
        return {
            "tanh_v":"np.tanh(x_velocity)",
            "ctrl_cost":"np.sum(np.square(action))"
        }