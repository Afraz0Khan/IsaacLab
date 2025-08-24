from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_32(RewardFunc):
    """
    Encourages upright posture via cosine of pitch.
    features: [x_vel, cos_pitch]
    """
    def __init__(self):
        self._w = np.array([1.2, 0.30])

    def __call__(self, env, state, action, next_state, x_velocity):
        if x_velocity is None: x_velocity=state[9]
        cos_pitch = np.cos(state[2])
        r = self._w[0]*x_velocity + self._w[1]*cos_pitch
        return r.item() if hasattr(r,"item") else r

    @property
    def weights(self): return self._w
    def set_weights(self,w): self._w=w
    def get_features(self):
        return {
            "x_vel":"x_velocity",
            "cos_pitch":"np.cos(state[2])"
        }