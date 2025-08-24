from typing import Optional, Dict
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class RewardFunc_2(RewardFunc):
    """
    Adds torso height bonus and small posture penalty.
    features: [x_velocity, ctrl_cost, torso_z, abs_pitch]
    """
    def __init__(self):
        self._w = np.array([1.5, -0.10, 0.50, -0.20])

    def __call__(self,
                 env: HalfCheetahEnv,
                 state: Optional[torch.Tensor],
                 action,
                 next_state,
                 x_velocity: Optional[float]) -> float:
        if x_velocity is None:
            x_velocity = state[9]
        ctrl_cost = np.sum(np.square(action))
        torso_z = env.get_body_com("torso")[2]
        abs_pitch = np.abs(state[2])
        r = (self._w[0]*x_velocity +
             self._w[1]*ctrl_cost +
             self._w[2]*torso_z +
             self._w[3]*abs_pitch)
        if hasattr(r, "item"): return r.item()
        return r

    @property
    def weights(self): return self._w
    def set_weights(self, w): self._w = w
    def get_features(self):
        return {
            "x_velocity": "x_velocity",
            "ctrl_cost": "np.sum(np.square(action))",
            "torso_z": "env.get_body_com('torso')[2]",
            "abs_pitch": "np.abs(state[2])"
        }