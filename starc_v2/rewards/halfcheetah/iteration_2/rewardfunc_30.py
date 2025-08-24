from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_30(RewardFunc):
    """
    Extra penalty on very large torques via squared ctrl term.
    features: [x_vel, ctrl_cost_sq]
    """
    def __init__(self):
        self._w = np.array([1.3, -0.005])

    def __call__(self, env, state, action, next_state, x_velocity):
        if x_velocity is None: x_velocity=state[9]
        ctrl_cost_sq = (np.sum(np.square(action)))**2
        r = self._w[0]*x_velocity + self._w[1]*ctrl_cost_sq
        return r.item() if hasattr(r,"item") else r

    @property
    def weights(self): return self._w
    def set_weights(self,w): self._w=w
    def get_features(self):
        return {
            "x_vel":"x_velocity",
            "ctrl_cost_sq":"(np.sum(np.square(action)))**2"
        }