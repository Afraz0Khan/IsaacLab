from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_23(RewardFunc):
    """
    Speed plus inverse torque for energetic efficiency.
    features: [x_vel, inv_sqrt_ctrl]
    """
    def __init__(self):
        self._w = np.array([1.0, 0.50])

    def __call__(self, env, state, action, next_state, x_velocity):
        if x_velocity is None: x_velocity=state[9]
        ctrl_cost = np.sum(np.square(action))
        inv_sqrt_ctrl = 1.0 / (1.0 + np.sqrt(ctrl_cost))
        r = self._w[0]*x_velocity + self._w[1]*inv_sqrt_ctrl
        return r.item() if hasattr(r,"item") else r

    @property
    def weights(self): return self._w
    def set_weights(self,w): self._w=w
    def get_features(self):
        return {
            "x_vel":"x_velocity",
            "inv_sqrt_ctrl":"1.0/(1.0+np.sqrt(np.sum(np.square(action))))"
        }