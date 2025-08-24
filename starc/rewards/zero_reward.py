# rewards/zero_reward.py
from typing import Optional
import torch

def zero_reward(env, state: Optional[torch.Tensor], action, next_state):
    return 0.0
