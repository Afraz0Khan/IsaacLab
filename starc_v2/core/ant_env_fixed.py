import numpy as np
from typing import Tuple, Callable, Any
from functools import partial
import sys
import os
import gym  # local import to avoid hard dependency at module level

# Import from the original starc directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'starc'))

from starc.core._types import EnvInfoCont, Space

class AntEnvFixed:
    """
    Environment wrapper for Ant that provides STARC compatibility.
    
    This wrapper creates a gym-like interface compatible with STARC analysis
    for Ant environments, handling the observation space and StateVals training.
    
    Key features:
    1. Tracks x_position for velocity calculation
    2. Passes x_velocity to reward functions that expect it
    3. Maintains compatibility with reward function signatures
    4. Provides proper observation space for StateVals training
    """

    # Ant observation space (based on standard Gymnasium Ant: 105 dimensions)
    # This is a simplified representation for STARC compatibility
    state_space: Space = [(-100, 100)] * 105  # 105 dimensions with reasonable bounds
    
    # Ant action space (8 dimensions for joint torques)
    act_space: Space = [(-1, 1)] * 8

    def __init__(self, reward_func, discount: float, n_episodes_sarsa: int = 10000, **kwargs):
        self.reward_func = reward_func
        self.reward_func_curried = partial(reward_func, self)
        self.discount = discount
        self.prev_obs = None
        
        # Track x_position for velocity calculation
        self.prev_x_position = None
        
        # Simulation timestep (typical for MuJoCo)
        self.dt = 0.02
        
        # Create mock gym spaces for compatibility
        self.observation_space = gym.spaces.Box(
            low=np.array([-100.0] * 105, dtype=np.float32),
            high=np.array([100.0] * 105, dtype=np.float32),
            dtype=np.float32,
        )
        
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0] * 8, dtype=np.float32),
            high=np.array([1.0] * 8, dtype=np.float32),
            dtype=np.float32,
        )

        # Set reward range for gym compatibility
        self.reward_range = (-float("inf"), float("inf"))
        
        # Optional SARSA value function for STARC analysis
        if n_episodes_sarsa and n_episodes_sarsa > 0:
            print(f"[INIT] AntEnvFixed: n_episodes_sarsa={n_episodes_sarsa}")
            from starc.core.state_vals import StateVals
            self.state_vals = StateVals(
                self,
                self.reward_func.__class__.__name__,
                n_episodes_sarsa,
            )
        else:
            self.state_vals = None  # type: ignore
        
        # Utility class for STARC compatibility
        self.env_info = EnvInfoCont(
            trans_dist=AntEnvFixed.predict_next_state,
            trans_dist_deterministic=True,
            discount=self.discount,
            state_space=AntEnvFixed.state_space,
            action_space=AntEnvFixed.act_space,
            state_vals=self.state_vals,  # type: ignore
            state_vals_deterministic=True,
        )

    def step(self, action, **kwargs):
        """Perform one environment step (Gymnasium API)."""
        
        # Store x_position before step (mock for STARC compatibility)
        x_position_before = self.prev_x_position if self.prev_x_position is not None else 0.0
        
        # Generate mock next observation (in practice, this would come from actual env)
        obs = self._generate_mock_observation()
        
        # Mock x-velocity calculation
        x_position_after = x_position_before + np.random.normal(0, 0.1)  # Mock forward progress
        x_velocity = (x_position_after - x_position_before) / self.dt
        
        # Store for next call
        self.prev_x_position = x_position_after

        # Compute custom reward
        reward = self._call_reward_function(action, obs, x_velocity)

        # Update previous observation reference
        self.prev_obs = obs

        # Mock termination conditions
        terminated = False
        truncated = False
        info = {}

        return obs.astype(np.float32), float(reward), terminated, truncated, info
    
    def _generate_mock_observation(self) -> np.ndarray:
        """Generate a mock observation for STARC compatibility."""
        # In practice, this would be replaced by actual environment observation
        return np.random.normal(0, 1, size=(105,)).astype(np.float32)
    
    def _call_reward_function(self, action, obs, x_velocity):
        """
        Call the reward function with the appropriate signature.
        
        Handles both old signature (env, prev_obs, action, obs) and 
        new signature (env, state, action, next_state, x_velocity, contact_forces)
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
        has_cf = 'contact_forces' in param_names

        # Mock contact forces (zeros for Ant)
        contact_forces = np.zeros(4, dtype=np.float32)  # 4 feet

        if expects_env:
            # Signatures with explicit env first
            if has_xvel and has_cf:
                # (env, state, action, next_state, x_velocity, contact_forces)
                return self.reward_func(self, prev_state, action, obs, x_velocity, contact_forces)
            elif has_xvel:
                # (env, state, action, next_state, x_velocity)
                return self.reward_func(self, prev_state, action, obs, x_velocity)
            elif has_cf:
                # (env, state, action, next_state, contact_forces)
                return self.reward_func(self, prev_state, action, obs, contact_forces)
            else:
                # (env, prev_obs, action, obs)
                return self.reward_func(self, prev_state, action, obs)
        else:
            # LLM-style signatures without env
            if has_xvel and has_cf:
                # (state, action, next_state, x_velocity, contact_forces)
                return self.reward_func(prev_state, action, obs, x_velocity, contact_forces)
            elif has_xvel:
                # (state, action, next_state, x_velocity)
                return self.reward_func(prev_state, action, obs, x_velocity)
            elif has_cf:
                # (state, action, next_state, contact_forces)
                return self.reward_func(prev_state, action, obs, contact_forces)
            else:
                # (prev_obs, action, obs)
                return self.reward_func(prev_state, action, obs)
    
    def reset(self, **kwargs):
        """Reset the environment (Gymnasium API)."""
        
        # Generate initial observation
        obs = self._generate_mock_observation()
        
        # Reset position tracking
        self.prev_x_position = 0.0
        self.prev_obs = None

        return obs.astype(np.float32), {}
    
    @staticmethod
    def predict_next_state(state, action):
        """Predict next state given current state and action (for STARC compatibility)."""
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if not isinstance(action, np.ndarray):
            action = np.array(action)

        # Simple mock prediction (in practice, this would use actual dynamics)
        # Add small random noise to simulate state transition
        next_state = state + np.random.normal(0, 0.01, size=state.shape)
        
        return next_state.astype(np.float32)
    
    @staticmethod
    def get_state_space():
        return AntEnvFixed.state_space
    
    @staticmethod
    def get_action_space():
        return AntEnvFixed.act_space
    
    @staticmethod
    def get_env_info():
        # Create a temporary instance to get env_info
        from starc.rewards.ground_truth_reward import GroundTruthReward
        temp_env = AntEnvFixed(GroundTruthReward(), 0.99, 0)
        return temp_env.env_info

    # Mock data attribute for compatibility with reward functions that access env.data
    @property
    def data(self):
        """Mock data attribute for compatibility."""
        class MockData:
            def __init__(self):
                # Mock contact forces (cfrc_ext) - 4 feet x 6 DOF each = 24 elements
                self.cfrc_ext = np.zeros(24, dtype=np.float32)
        
        return MockData()
