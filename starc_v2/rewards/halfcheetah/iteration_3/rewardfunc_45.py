from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_45(RewardFunc):
    """
    Penalises torque via sigmoid to soften small torques.
    features: [x_vel, torque_sigmoid]
    """
    def __init__(self):
        self._w = np.array([1.25, -1.0])

    def __call__(self, env, state, action, next_state, x_velocity):
        if x_velocity is None: x_velocity=state[9]
        torque_mag = np.sum(np.abs(action))
        torque_sigmoid = 1.0/(1.0+np.exp(-torque_mag))  # ∈ (0,1)
        r = self._w[0]*x_velocity + self._w[1]*torque_sigmoid
        return r.item() if hasattr(r,"item") else r

    @property
    def weights(self): return self._w
    def set_weights(self,w): self._w=w
    def get_features(self):
        return {
            "x_vel":"x_velocity",
            "torque_sigmoid":"1/(1+np.exp(-np.sum(np.abs(action))))"
        }