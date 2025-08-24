from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_4(RewardFunc):
    """
    Promotes speed while discouraging high angular velocities.
    features: [x_velocity, ctrl_cost, ang_vel_sq]
    """
    def __init__(self):
        self._w = np.array([1.4, -0.05, -0.01])

    def __call__(self, env, state, action, next_state, x_velocity):
        if x_velocity is None:
            x_velocity = state[9]
        ctrl_cost = np.sum(np.square(action))
        ang_vel_sq = np.sum(np.square(state[11:]))   # all angular vels
        r = self._w[0]*x_velocity + self._w[1]*ctrl_cost + self._w[2]*ang_vel_sq
        if hasattr(r,"item"): return r.item()
        return r

    @property
    def weights(self): return self._w
    def set_weights(self, w): self._w=w
    def get_features(self):
        return {
            "x_velocity": "x_velocity",
            "ctrl_cost": "np.sum(np.square(action))",
            "ang_vel_sq": "np.sum(np.square(state[11:]))"
        }