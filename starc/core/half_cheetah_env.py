import numpy as np
from typing import Tuple, Callable, Any
from functools import partial
import sys
import os

from custom_gym_envs.envs.half_cheetah.HalfCheetah_V1.half_cheetah import HalfCheetahEnv as OriginalHalfCheetah
from ._types import EnvInfoCont, Space



class HalfCheetahEnv(OriginalHalfCheetah):
    """
    A customized version of the half cheetah env allowing you to pass in a custom
    reward function
    """
    # Remove the shared instance - this was causing thread safety issues
    # original_env_instance = OriginalHalfCheetah()

    state_space: Space = [
        # Position dimensions (excluding root x-position by default)
        (-100, 100),  # z-coordinate of the front tip (rootz)
        (-3.14, 3.14),  # angle of the front tip (rooty)
        (-3.14, 3.14),  # angle of the back thigh (bthigh)
        (-3.14, 3.14),  # angle of the back shin (bshin)
        (-3.14, 3.14),  # angle of the back foot (bfoot)
        (-3.14, 3.14),  # angle of the front thigh (fthigh)
        (-3.14, 3.14),  # angle of the front shin (fshin)
        (-3.14, 3.14),  # angle of the front foot (ffoot)
        
        # Velocity dimensions
        (-10, 10),  # velocity of the x-coordinate of front tip
        (-10, 10),  # velocity of the z-coordinate of front tip
        (-10, 10),  # angular velocity of the front tip
        (-10, 10),  # angular velocity of the back thigh
        (-10, 10),  # angular velocity of the back shin
        (-10, 10),  # angular velocity of the back foot
        (-10, 10),  # angular velocity of the front thigh
        (-10, 10),  # angular velocity of the front shin
        (-10, 10),  # angular velocity of the front foot
    ]
    
    # Define action space (6 dimensions for controlling joints)
    act_space: Space = [
        (-1, 1),  # Torque applied on the back thigh rotor
        (-1, 1),  # Torque applied on the back shin rotor
        (-1, 1),  # Torque applied on the back foot rotor
        (-1, 1),  # Torque applied on the front thigh rotor
        (-1, 1),  # Torque applied on the front shin rotor
        (-1, 1),  # Torque applied on the front foot rotor
    ]

    def __init__(self, reward_func, discount: float, n_episodes_sarsa: int = 10000, **kwargs):
        self.reward_func = reward_func
        self.reward_func_curried = partial(reward_func, self)
        self.discount = discount
        self.prev_obs = None
        super().__init__(**kwargs)

        from .state_vals import StateVals
        self.state_vals = StateVals(self,
                                    self.reward_func.__class__.__name__,
                                    n_episodes_sarsa)
        
        # utility class allowing us to pass around the information about the env easily
        self.env_info = EnvInfoCont(
            trans_dist=HalfCheetahEnv.predict_next_state,
            trans_dist_deterministic=True,
            discount=self.discount,
            state_space=HalfCheetahEnv.state_space,
            action_space=HalfCheetahEnv.act_space,
            state_vals=self.state_vals,
            state_vals_deterministic=True,
        )

    def step(self, *args, **kwargs) -> Tuple:
        step_result = super().step(*args, **kwargs)
        if len(step_result) == 5:
            obs, _, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, _, done, info = step_result
        
        reward = self.reward_func(self, self.prev_obs, args[0], obs)
        self.prev_obs = obs

        return obs, reward, done, info
    
    @staticmethod
    def predict_next_state(state, action):
        # if the arrays aren't already numpy arrays, convert them
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if not isinstance(action, np.ndarray):
            action = np.array(action)

        # Create a new environment instance for thread safety
        env = OriginalHalfCheetah()

        # HalfCheetah observation excludes x-position, so we have:
        # - 8 position elements (excluding x)
        # - 9 velocity elements
        # Total: 17 elements
        
        # Extract position (without x) and velocity from the 17-dim observation
        position_no_x = state[:8]  # First 8 elements are position without x
        velocity = state[8:17]     # Next 9 elements are velocity
        
        # Reconstruct full position by adding x-coordinate (we'll use 0 as default)
        # In practice, x-position doesn't affect the dynamics for prediction
        x_position = 0.0
        full_position = np.concatenate([[x_position], position_no_x])
        
        # Set environment to desired state
        env.set_state(full_position, velocity)
        
        # Take the desired action
        step_result = env.step(action)
        if len(step_result) == 5:
            next_state, _, _, _, _ = step_result
        else:
            next_state, _, _, _ = step_result
        
        return next_state
    
    @staticmethod
    def get_state_space():
        return HalfCheetahEnv.state_space
    
    @staticmethod
    def get_action_space():
        return HalfCheetahEnv.act_space
    
    @staticmethod
    def get_env_info():
        return HalfCheetahEnv.env_info
    