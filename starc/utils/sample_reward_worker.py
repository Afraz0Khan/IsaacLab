import numpy as np
from starc.core._types import RewardCont, TransCont, Space
import traceback

def sample_reward_worker_safe(reward: RewardCont,
                         trans_dist: TransCont,
                         state_space: Space,
                         action_space: Space):
    """
    A safer version of sample_reward_worker that handles exceptions in MuJoCo.
    
    Args:
        reward: The reward function to evaluate
        trans_dist: The transition function
        state_space: The state space bounds
        action_space: The action space bounds
        
    Returns:
        The reward value or 0.0 if an error occurs
    """
    try:
        # Sample state and action
        s = [np.random.uniform(*interval) for interval in state_space]
        a = [np.random.uniform(*interval) for interval in action_space]
        
        # Get next state through transition
        try:
            s_prime = trans_dist(s, a)
            
            # Evaluate reward
            return reward(s, a, s_prime)
        except Exception as e:
            # If transition fails, return default value
            print(f"Warning: Transition error: {str(e)[:100]}...")
            return 0.0
            
    except Exception as e:
        # Catch any other errors
        print(f"Error in sample_reward_worker: {str(e)[:100]}...")
        return 0.0 