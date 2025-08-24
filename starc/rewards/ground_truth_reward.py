from typing import Optional
import numpy as np
import torch
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.core.reward_func import RewardFunc

class GroundTruthReward(RewardFunc):
    """
        This is the original reward function from the HalfCheetahEnv class.
    """
    def __call__(self,
                 env: HalfCheetahEnv,
                 state: Optional[torch.Tensor], #TODO fix the types
                 action,
                 next_state,
                 x_velocity: None) -> float:
        # Original HalfCheetah reward function:
        # reward = forward_reward - ctrl_cost
        # x_velocity is calculated as follows:
        # x_position_before = env.data.qpos[0]
        # env.do_simulation(action, env.frame_skip)
        # x_position_after = env.data.qpos[0]
        # x_velocity = (x_position_after - x_position_before) / env.dt

        forward_reward = 1.0 * x_velocity
        ctrl_cost = 0.1 * np.sum(np.square(action))
        reward = forward_reward - ctrl_cost

        if hasattr(reward, 'item'): return reward.item()
        return reward