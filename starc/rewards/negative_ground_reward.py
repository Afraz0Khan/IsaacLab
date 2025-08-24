from typing import Optional
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc
from starc.rewards.ground_truth_reward import GroundTruthReward

class NegativeGroundReward(RewardFunc):
    """
        This is the original reward function from the HalfCheetahEnv class.
    """
    def __init__(self):
        self.ground_truth_reward = GroundTruthReward()

    def __call__(self,
                 env: HalfCheetahEnv,
                 state: Optional[torch.Tensor],
                 action,
                 next_state) -> float:
        return -self.ground_truth_reward(env, state, action, next_state)
