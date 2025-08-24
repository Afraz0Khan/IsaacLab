import numpy as np
from typing import Tuple


def build_transition_batch(env_name: str, n_samples: int = 96, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a transition batch (S, A, SP, X_VEL) for the requested env using Gymnasium Mujoco tasks.

    - env_name: 'halfcheetah' | 'ant' | 'humanoid'
    - returns: S (Nxd), A (Nxu), SP (Nxd), X_VEL (Nx)
    """
    try:
        import gymnasium as gym
    except Exception as e:
        raise RuntimeError("Gymnasium is required to build transitions for non-HalfCheetah envs") from e

    env_id = {
        'halfcheetah': 'HalfCheetah-v4',
        'ant': 'Ant-v4',
        'humanoid': 'Humanoid-v4',
    }.get(env_name, 'HalfCheetah-v4')

    env = gym.make(env_id)
    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()

    rng = np.random.default_rng(seed)
    dt = getattr(getattr(env, 'unwrapped', env), 'dt', 0.02)

    S_list, A_list, SP_list, XV_list = [], [], [], []

    # Try to read root x from mujoco data when available
    def get_root_x(_env):
        try:
            return float(_env.unwrapped.data.qpos[0])
        except Exception:
            return None

    # Sample random transitions
    obs, _ = env.reset()
    prev_root_x = get_root_x(env)
    for _ in range(n_samples):
        # Random action within bounds
        if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
            low, high = env.action_space.low, env.action_space.high
            act = rng.uniform(low, high).astype(np.float32)
        else:
            act = rng.standard_normal(size=(env.action_space.shape or (1,))[0]).astype(np.float32)

        # Step
        root_x_before = get_root_x(env)
        res = env.step(act)
        if len(res) == 5:
            next_obs, _, terminated, truncated, info = res
            done = terminated or truncated
        else:
            next_obs, _, done, info = res
        root_x_after = get_root_x(env)

        # x_velocity
        if root_x_before is not None and root_x_after is not None:
            x_vel = (root_x_after - root_x_before) / float(dt or 0.02)
        else:
            # Fallback: 0 if root position unavailable
            x_vel = 0.0

        S_list.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        A_list.append(act.reshape(-1))
        SP_list.append(np.asarray(next_obs, dtype=np.float32).reshape(-1))
        XV_list.append(float(x_vel))

        obs = next_obs
        if done:
            obs, _ = env.reset()

    env.close()

    return (
        np.vstack(S_list),
        np.vstack(A_list),
        np.vstack(SP_list),
        np.asarray(XV_list, dtype=np.float32),
    )

