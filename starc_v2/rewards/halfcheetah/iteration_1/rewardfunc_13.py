from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_13(RewardFunc):
    """
    Uses sqrt(ctrl) for a milder high-torque penalty.
    features: [x_velocity, sqrt_ctrl]
    """
    def __init__(self):
        self._w = np.array([1.3, -0.15])

    def __call__(self, env, state, action, next_state, x_velocity):
        if x_velocity is None: x_velocity = state[9]
        sqrt_ctrl = np.sqrt(np.sum(np.square(action)))
        r = self._w[0]*x_velocity + self._w[1]*sqrt_ctrl
        if hasattr(r,'item'): return r.item()
        return r

    @property
    def weights(self): return self._w
    def set_weights(self,w): self._w=w
    def get_features(self):
        return {
            "x_velocity":"x_velocity",
            "sqrt_ctrl":"np.sqrt(np.sum(np.square(action)))"
        }