from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_27(RewardFunc):
    """
    High torso height & low pitch bonus.
    features: [x_vel, torso_z, abs_pitch, ctrl_cost]
    """
    def __init__(self):
        self._w = np.array([1.15, 0.50, -0.20, -0.05])

    def __call__(self, env, state, action, next_state, x_velocity):
        if x_velocity is None: x_velocity=state[9]
        torso_z = env.get_body_com("torso")[2]
        abs_pitch = np.abs(state[2])
        ctrl_cost = np.sum(np.square(action))
        r = (self._w[0]*x_velocity + self._w[1]*torso_z +
             self._w[2]*abs_pitch + self._w[3]*ctrl_cost)
        return r.item() if hasattr(r,"item") else r

    @property
    def weights(self): return self._w
    def set_weights(self,w): self._w=w
    def get_features(self):
        return {
            "x_vel":"x_velocity",
            "torso_z":"env.get_body_com('torso')[2]",
            "abs_pitch":"np.abs(state[2])",
            "ctrl_cost":"np.sum(np.square(action))"
        }