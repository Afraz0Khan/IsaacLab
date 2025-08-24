#!/usr/bin/env python3
"""
Compute Euclidean (L2) distances between three reward functions in the Half-Cheetah
state–action space using the same machinery that STARC employs.

• GroundTruthReward          ("EXPERT")
• NegativeGroundRewardFixed  ("-EXPERT")
• NegForwardReward           (-forwardVel – ctrlCost)

The script:
1. Samples the canonical transition batch with STARCv2Analyzer (same N_SAMPLES
   and discount) from the ground-truth environment.
2. Builds reward vectors for the three rewards (unit-norm, like STARC).
3. Prints the pairwise L2 distances.

Run from project root:
    python starc_v2/scripts/compare_reward_distances.py
"""

import numpy as np
import pathlib

from starc_v2.core.half_cheetah_env_fixed import HalfCheetahEnvFixed
from starc_v2.core.starc_analyzer import STARCv2Analyzer
from starc_v2.config import STARCv2Config

from starc.rewards.ground_truth_reward import GroundTruthReward
from starc_v2.rewards.negative_ground_reward_fixed import NegativeGroundRewardFixed

# ---------------------------------------------------------------------------
# Custom reward:  -forward_velocity  - ctrl_cost
# ---------------------------------------------------------------------------
from typing import Optional
import numpy as np
import torch
from starc.core.reward_func import RewardFunc
from starc.core.half_cheetah_env import HalfCheetahEnv

class NegForwardReward(RewardFunc):
    """Reward = -1 * x_velocity  - 0.1 * sum(action^2)."""
    def __call__(
        self,
        env: HalfCheetahEnv,
        state: Optional[torch.Tensor],
        action,
        next_state,
        x_velocity: float = None,
    ) -> float:
        forward_term = -1.0 * x_velocity
        ctrl_cost = 0.1 * np.sum(np.square(action))
        reward = forward_term - ctrl_cost
        return reward.item() if hasattr(reward, "item") else reward

# ---------------------------------------------------------------------------
# Helper to compute normalised reward vector using STARC internals
# ---------------------------------------------------------------------------

def reward_vector(reward_obj, analyzer: STARCv2Analyzer, S, A, SP, X_VEL):
    env = HalfCheetahEnvFixed(reward_obj, analyzer.discount, analyzer.n_episodes_sarsa)
    vec = analyzer._compute_reward_vector(env, S, A, SP, X_VEL)  # pylint: disable=protected-access
    return vec


def main():
    analyzer = STARCv2Analyzer()

    # Build a transition batch using GroundTruthReward env
    dummy_env = HalfCheetahEnvFixed(GroundTruthReward(), analyzer.discount, analyzer.n_episodes_sarsa)
    S, A, SP, X_VEL = analyzer._build_transition_batch(dummy_env)  # pylint: disable=protected-access

    rewards = {
        "EXPERT": GroundTruthReward(),
        "-EXPERT": NegativeGroundRewardFixed(),
        "-forwardVel-ctrlCost": NegForwardReward(),
    }

    vectors = {name: reward_vector(obj, analyzer, S, A, SP, X_VEL) for name, obj in rewards.items()}

    names = list(vectors.keys())
    print("\nPairwise L2 distances (unit-norm reward vectors):\n")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = np.linalg.norm(vectors[names[i]] - vectors[names[j]])
            print(f"{names[i]:<20} ↔ {names[j]:<20} : {d:.4f}")


if __name__ == "__main__":
    main() 