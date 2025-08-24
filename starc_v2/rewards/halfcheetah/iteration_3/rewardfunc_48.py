from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_48(RewardFunc):
    """
    Square-root speed reward, quadratic torque cost for high-torque suppression.
    features: [sqrt_speed, ctrl_cost_sq]
    """
    def __init__(self):
        self._w = np.array([2.0, -0.01])

    def __call__(self, env, state, action, next_state, x_velocity):
        if x_velocity is None: x_velocity=state[9]
        sqrt_speed = np.sign(x_velocity) * np.sqrt(np.abs(x_velocity))
        ctrl_cost_sq = (np.sum(np.square(action)))**2
        r = self._w[0]*sqrt_speed + self._w[1]*ctrl_cost_sq
        return r.item() if hasattr(r,"item") else r

    @property
    def weights(self): return self._w
    def set_weights(self,w): self._w=w
    def get_features(self):
        return {
            "sqrt_speed":"np.sign(x_velocity)*np.sqrt(np.abs(x_velocity))",
            "ctrl_cost_sq":"(np.sum(np.square(action)))**2"
        }