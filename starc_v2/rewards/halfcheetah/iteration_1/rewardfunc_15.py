from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_15(RewardFunc):
    """
    Encourages symmetric leg positions (back vs front thigh).
    features: [x_velocity, ctrl_cost, thigh_sym_pen]
    """
    def __init__(self):
        self._w = np.array([1.2, -0.05, -0.20])

    def __call__(self, env, state, action, next_state, x_velocity):
        if x_velocity is None: x_velocity = state[9]
        ctrl_cost = np.sum(np.square(action))
        thigh_sym_pen = (state[3] - state[6])**2
        r = self._w[0]*x_velocity + self._w[1]*ctrl_cost + self._w[2]*thigh_sym_pen
        if hasattr(r,'item'): return r.item()
        return r

    @property
    def weights(self): return self._w
    def set_weights(self,w): self._w=w
    def get_features(self):
        return {
            "x_velocity":"x_velocity",
            "ctrl_cost":"np.sum(np.square(action))",
            "thigh_sym_pen":"(state[3]-state[6])**2"
        }