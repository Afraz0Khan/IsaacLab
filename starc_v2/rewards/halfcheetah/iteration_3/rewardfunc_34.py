from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_34(RewardFunc):
    """
    Log-speed bonus, L1 torque cost, front-foot stability.
    features: [log_speed, l1_torque, abs_front_foot_angle]
    """
    def __init__(self):
        self._w = np.array([1.4, -0.4, -0.15])

    def __call__(self, env, state, action, next_state, x_velocity):
        if x_velocity is None: x_velocity = state[9]
        log_speed = np.log1p(np.abs(x_velocity)) * np.sign(x_velocity)
        l1_torque = np.sum(np.abs(action))
        abs_front_foot_angle = np.abs(state[8])
        r = (self._w[0]*log_speed +
             self._w[1]*l1_torque +
             self._w[2]*abs_front_foot_angle)
        return r.item() if hasattr(r,"item") else r

    @property
    def weights(self): return self._w
    def set_weights(self, w): self._w = w
    def get_features(self):
        return {
            "log_speed": "np.log1p(np.abs(x_velocity))*np.sign(x_velocity)",
            "l1_torque": "np.sum(np.abs(action))",
            "abs_front_foot_angle": "np.abs(state[8])"
        }