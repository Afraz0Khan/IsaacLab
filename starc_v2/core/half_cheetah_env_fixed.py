import numpy as np
from typing import Tuple, Callable, Any
from functools import partial
import sys
import os
import gym  # local import to avoid hard dependency at module level

# Import from the original starc directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'starc'))

from custom_gym_envs.envs.half_cheetah.HalfCheetah_V1.half_cheetah import HalfCheetahEnv as OriginalHalfCheetah
from starc.core._types import EnvInfoCont, Space

class HalfCheetahEnvFixed(OriginalHalfCheetah):
    """
    Fixed version of the STARC HalfCheetahEnv that properly computes x_velocity
    for reward functions like GroundTruthReward that need it.
    
    Key fixes:
    1. Tracks x_position before and after step to compute x_velocity
    2. Passes x_velocity to reward functions that expect it
    3. Maintains compatibility with both old and new reward function signatures
    """

    state_space: Space = [
        # Position dimensions (excluding root x-position by default)
        (-100, 100),  # x-coordinate of the front tip (rootx)
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
        
        # Track x_position for velocity calculation
        self.prev_x_position = None
        
        super().__init__(**kwargs)

        # ------------------------------------------------------------------
        # Gym wrappers (e.g. TimeLimit) expect the environment to expose a
        # *reward_range* attribute.  The original MuJoCo Half-Cheetah sets
        # this but our subclass overrides __init__ without re-defining it,
        # which breaks wrappers.  Provide an open-ended default so external
        # code can still query it.
        # ------------------------------------------------------------------
        self.reward_range = (-float("inf"), float("inf"))
        
        # ------------------------------------------------------------------
        # SB3 shimmy cannot convert spaces with infinite bounds. If the
        # underlying MuJoCo Half-Cheetah exposes Box(-inf, inf) we replace
        # those infinities by large finite sentinels so the compatibility
        # layer accepts them.
        # ------------------------------------------------------------------
        import numpy as _np

        if isinstance(self.observation_space, gym.spaces.Box):
            low, high = self.observation_space.low, self.observation_space.high

            # Replace infinities for shimmy / gymnasium compatibility
            if _np.isinf(low).any() or _np.isinf(high).any():
                low = _np.where(_np.isinf(low), -1e10, low)
                high = _np.where(_np.isinf(high), 1e10, high)

            # Cast to float32 as required by gymnasium best-practices
            self.observation_space = gym.spaces.Box(
                low=low.astype(_np.float32),
                high=high.astype(_np.float32),
                dtype=_np.float32,
            )

        # Ensure action space dtype is float32 for consistency
        if isinstance(self.action_space, gym.spaces.Box):
            self.action_space = gym.spaces.Box(
                low=self.action_space.low.astype(_np.float32),
                high=self.action_space.high.astype(_np.float32),
                dtype=_np.float32,
            )
        
        # --------------------------------------------------------------
        # Optional SARSA value function (used by STARC analysis but not
        # required for every use-case, e.g. GA fitness evaluation).  When
        # *n_episodes_sarsa* is 0 or negative we skip the expensive
        # training entirely.
        # --------------------------------------------------------------
        if n_episodes_sarsa and n_episodes_sarsa > 0:
            print(f"[INIT] HalfCheetahEnvFixed: n_episodes_sarsa={n_episodes_sarsa}")
            from starc.core.state_vals import StateVals
            self.state_vals = StateVals(
                self,
                self.reward_func.__class__.__name__,
                n_episodes_sarsa,
            )
        else:
            self.state_vals = None  # type: ignore
        
        # utility class allowing us to pass around the information about the env easily
        self.env_info = EnvInfoCont(
            trans_dist=HalfCheetahEnvFixed.predict_next_state,
            trans_dist_deterministic=True,
            discount=self.discount,
            state_space=HalfCheetahEnvFixed.state_space,
            action_space=HalfCheetahEnvFixed.act_space,
            state_vals=self.state_vals,
            state_vals_deterministic=True,
        )

    # ------------------------------------------------------------------
    # Gymnasium-style step implementation
    # ------------------------------------------------------------------
    def step(self, action, **kwargs):
        """Perform one environment step (Gymnasium API).

        Returns
        -------
        obs : np.ndarray
        reward : float
        terminated : bool
            True if episode finished due to task completion / failure.
        truncated : bool
            True if episode was interrupted by time limit or external signal.
        info : dict
        """

        # Store x_position before step
        x_position_before = self.data.qpos[0] if hasattr(self, 'data') else 0.0

        # Parent step (MuJoCo env) – may return 4-tuple or 5-tuple
        parent_res = super().step(action, **kwargs)

        if len(parent_res) == 5:
            obs, _, terminated, truncated, info = parent_res
        else:  # legacy 4-tuple
            obs, _, done, info = parent_res
            terminated, truncated = bool(done), False

        # Compute x-velocity
        x_position_after = self.data.qpos[0] if hasattr(self, 'data') else 0.0
        x_velocity = (x_position_after - x_position_before) / self.dt

        # Store for next call
        self.prev_x_position = x_position_after

        # Compute custom reward
        reward = self._call_reward_function(action, obs, x_velocity)

        # Update previous observation reference
        self.prev_obs = obs

        return obs.astype(np.float32), float(reward), terminated, truncated, info
    
    def _call_reward_function(self, action, obs, x_velocity):
        """
        Call the reward function with the appropriate signature.
        
        Handles both old signature (env, prev_obs, action, obs) and 
        new signature (env, state, action, next_state, x_velocity)
        """
        import inspect
        
        # Get the signature of the reward function
        if hasattr(self.reward_func, '__call__'):
            sig = inspect.signature(self.reward_func.__call__)
        else:
            sig = inspect.signature(self.reward_func)
        
        param_names = list(sig.parameters.keys())
        
        # Provide a safe default for the first step when self.prev_obs is None
        prev_state = self.prev_obs if self.prev_obs is not None else np.zeros_like(obs)

        # Decide whether reward expects explicit env argument
        expects_env = len(param_names) > 0 and param_names[0] == 'env'

        has_xvel = 'x_velocity' in param_names

        if expects_env:
            # Signatures with explicit env first
            if has_xvel:
                # (env, state, action, next_state, x_velocity)
                return self.reward_func(self, prev_state, action, obs, x_velocity)
            else:
                # (env, prev_obs, action, obs)
                return self.reward_func(self, prev_state, action, obs)
        else:
            # LLM-style signatures without env
            if has_xvel:
                # (state, action, next_state, x_velocity)
                return self.reward_func(prev_state, action, obs, x_velocity)
            else:
                # (prev_obs, action, obs)
                return self.reward_func(prev_state, action, obs)
    
    # ------------------------------------------------------------------
    # Gymnasium-style reset implementation
    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        """Reset the environment (Gymnasium API)."""

        parent_res = super().reset(**kwargs)

        if isinstance(parent_res, tuple):
            obs, info = parent_res
        else:
            obs, info = parent_res, {}

        # Track initial x position
        if hasattr(self, 'data'):
            self.prev_x_position = self.data.qpos[0]
        else:
            self.prev_x_position = 0.0

        self.prev_obs = None

        return obs.astype(np.float32), info
    
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
        return HalfCheetahEnvFixed.state_space
    
    @staticmethod
    def get_action_space():
        return HalfCheetahEnvFixed.act_space
    
    @staticmethod
    def get_env_info():
        return HalfCheetahEnvFixed.env_info 