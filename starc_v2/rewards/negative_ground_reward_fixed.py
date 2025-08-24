from typing import Optional
import numpy as np
import torch
import sys
import os

# Add path to import from original starc
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'starc'))

from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc
from starc.rewards.ground_truth_reward import GroundTruthReward

class NegativeGroundRewardFixed(RewardFunc):
    """
    Fixed version of NegativeGroundReward that properly handles x_velocity parameter.
    
    This is the negative of the ground truth reward function, properly passing
    the x_velocity parameter to the underlying GroundTruthReward.
    """
    def __init__(self):
        self.ground_truth_reward = GroundTruthReward()

    def __call__(self,
                 env: HalfCheetahEnv,
                 state: Optional[torch.Tensor],
                 action,
                 next_state,
                 x_velocity: float = None) -> float:
        """
        Compute negative ground truth reward.
        
        Args:
            env: HalfCheetah environment
            state: Previous state
            action: Action taken
            next_state: Next state
            x_velocity: X-velocity of the robot (required for ground truth calculation)
        
        Returns:
            Negative of the ground truth reward
        """
        return -self.ground_truth_reward(env, state, action, next_state, x_velocity) 