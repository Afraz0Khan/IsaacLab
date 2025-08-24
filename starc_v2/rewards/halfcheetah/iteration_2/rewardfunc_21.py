from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_21(RewardFunc):
    """
    Rewards symmetric thigh movement via cosine of angle difference.
    features: [x_vel, cos_thigh_diff, ctrl_cost]
    """
    def __init__(self):
        self._w = np.array([1.2, 0.30, -0.05])

    def __call__(self, env, state, action, next_state, x_velocity):
        if x_velocity is None: x_velocity = state[9]
        cos_thigh_diff = np.cos(state[3]-state[6])
        ctrl_cost = np.sum(np.square(action))
        r = self._w[0]*x_velocity + self._w[1]*cos_thigh_diff + self._w[2]*ctrl_cost
        return r.item() if hasattr(r,"item") else r

    @property
    def weights(self): return self._w
    def set_weights(self,w): self._w=w
    def get_features(self):
        return {
            "x_vel":"x_velocity",
            "cos_thigh_diff":"np.cos(state[3]-state[6])",
            "ctrl_cost":"np.sum(np.square(action))"
        }