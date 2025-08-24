from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_31(RewardFunc):
    """
    Penalises abrupt changes between adjacent joint angles.
    features: [x_vel, angle_curvature]
    """
    def __init__(self):
        self._w = np.array([1.1, -0.10])

    def __call__(self, env, state, action, next_state, x_velocity):
        if x_velocity is None: x_velocity=state[9]
        angle_curvature = np.sum(np.square(np.diff(state[3:9])))
        r = self._w[0]*x_velocity + self._w[1]*angle_curvature
        return r.item() if hasattr(r,"item") else r

    @property
    def weights(self): return self._w
    def set_weights(self,w): self._w=w
    def get_features(self):
        return {
            "x_vel":"x_velocity",
            "angle_curvature":"np.sum(np.square(np.diff(state[3:9])))"
        }