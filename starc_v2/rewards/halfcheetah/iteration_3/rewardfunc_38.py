from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_38(RewardFunc):
    """
    Uses L1 torque cost (smoother penalty).
    features: [x_vel, l1_torque]
    """
    def __init__(self):
        self._w = np.array([1.25, -0.25])

    def __call__(self, env, state, action, next_state, x_velocity):
        if x_velocity is None: x_velocity = state[9]
        l1_torque = np.sum(np.abs(action))
        r = self._w[0]*x_velocity + self._w[1]*l1_torque
        return r.item() if hasattr(r,"item") else r

    @property
    def weights(self): return self._w
    def set_weights(self,w): self._w = w
    def get_features(self):
        return {
            "x_vel":"x_velocity",
            "l1_torque":"np.sum(np.abs(action))"
        }