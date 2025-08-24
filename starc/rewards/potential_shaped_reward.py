from typing import Optional
import numpy as np
import torch
from starc.core.reward_func import RewardFunc
from starc.rewards.ground_truth_reward import GroundTruthReward



class PotentialShapedReward(RewardFunc):
    """
    Same dense objective as the MuJoCo ground‑truth, plus a random linear
    potential Φ(s) = w·s + b.  The weights are sampled once at construction.
    """

    def __init__(self, gamma: float = 0.99, seed: Optional[int] = None):
        super().__init__()
        self.gt = GroundTruthReward()
        rng = np.random.default_rng(seed)

        # Half‑Cheetah observation space is (17,) by default
        self._w: Optional[np.ndarray] = None   # filled lazily after seeing env
        self._b: float = float(rng.random())
        self.gamma: float = gamma
        self.rng = rng

    # ------------- private helpers -------------
    def _init_weights_if_needed(self, obs_dim: int):
        if self._w is None:
            self._w = self.rng.normal(size=obs_dim)

    def _phi(self, state: np.ndarray) -> float:
        return float(np.dot(self._w, state) + self._b)

    # ------------- public API ------------------
    def __call__(
        self,
        env,                                      # HalfCheetah env
        state: Optional[torch.Tensor],
        action,
        next_state,
    ) -> float:

        # Lazy weight initialisation using env.observation_space shape
        self._init_weights_if_needed(env.observation_space.shape[0])

        # Base task reward
        r = self.gt(env, state, action, next_state)

        # Potential‑based shaping  γ Φ(s') − Φ(s)
        if state is not None:
            s_np        = state.cpu().numpy() if torch.is_tensor(state) else state
            s_next_np   = (
                next_state.cpu().numpy() if torch.is_tensor(next_state) else next_state
            )

            r += self.gamma * self._phi(s_next_np) - self._phi(s_np)

        return float(r)
