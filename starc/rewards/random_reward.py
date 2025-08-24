from typing import Optional
import numpy as np
import torch
from starc.core.reward_func import RewardFunc
from starc.core.half_cheetah_env import HalfCheetahEnv


class RandomReward(RewardFunc):
    """
    R_rand(s, a, s') =  w_s · s  +  w_a · a  +  w_sp · s'  +  b
    All weights are sampled once at construction time from N(0,1).
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        rng = np.random.default_rng(seed)

        # Hard‑coded dims for HalfCheetah‑v4 (obs=17, act=6)
        self.w_s   = rng.normal(size=17)
        self.w_a   = rng.normal(size=6)
        self.w_sp  = rng.normal(size=17)
        self.bias  = float(rng.random())

    # ----------------------------------------------------------
    def __call__(
        self,
        env: HalfCheetahEnv,
        state: Optional[torch.Tensor],
        action,
        next_state,
    ) -> float:

        if state is None:      # can happen at first reset
            return 0.0

        s_np   = state.cpu().numpy()      if torch.is_tensor(state) else state
        a_np   = action.cpu().numpy()     if torch.is_tensor(action) else action
        sp_np  = next_state.cpu().numpy() if torch.is_tensor(next_state) else next_state

        reward = (
            np.dot(self.w_s,  s_np)
            + np.dot(self.w_a, a_np)
            + np.dot(self.w_sp, sp_np)
            + self.bias
        )
        return float(reward)
