from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_22(RewardFunc):
    """
    Penalises extreme back-foot angles.
    features: [x_vel, abs_back_foot_angle, ctrl_cost]
    """
    def __init__(self):
        self._w = np.array([1.3, -0.20, -0.04])

    def __call__(self, env, state, action, next_state, x_velocity):
        if x_velocity is None: x_velocity=state[9]
        abs_back_foot_angle = np.abs(state[5])
        ctrl_cost = np.sum(np.square(action))
        r = self._w[0]*x_velocity + self._w[1]*abs_back_foot_angle + self._w[2]*ctrl_cost
        return r.item() if hasattr(r,"item") else r

    @property
    def weights(self): return self._w
    def set_weights(self,w): self._w=w
    def get_features(self):
        return {
            "x_vel":"x_velocity",
            "abs_back_foot_angle":"np.abs(state[5])",
            "ctrl_cost":"np.sum(np.square(action))"
        }