# scripts/build_dbase.py
import numpy as np
import os

from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.utils.simple_utils import sample_space
from starc.rewards.zero_reward import zero_reward

N = 60_000                              # or 8_000 Sobol points
DISCOUNT = 0.99                         # any value, irrelevant for physics
SAVE_PATH = "data/D_base.npy"

def main():
    # Create env with dummy reward (reward not used here)
    env = HalfCheetahEnv(zero_reward, DISCOUNT, n_episodes_sarsa=1)
    transitions = []

    obs, _ = env.reset()
    for i in range(N):
        action = np.array(sample_space(HalfCheetahEnv.act_space),
                          dtype=np.float32)
        s      = obs.copy()
        obs, *_ = env.step(action)[:4]   # ignore reward/done/info
        s_p    = obs.copy()
        transitions.append(np.hstack([s, action, s_p]))

        # optional: reset occasionally to diversify positions
        if i % 100 == 0:
            obs, _ = env.reset()

    transitions = np.vstack(transitions).astype(np.float32)
    os.makedirs("data", exist_ok=True)
    np.save(SAVE_PATH, transitions)
    print("saved", transitions.shape, "to", SAVE_PATH)

if __name__ == "__main__":
    main()