import numpy as np
from functools import wraps, partial
import time
from starc.core._types import Space, RewardCont
from starc.algorithms.distance import canon_and_norm_cont
from starc.utils.simple_utils import timed, sample_space, softmax
from typing import Callable

def _reward_diff(r1: RewardCont, r2: RewardCont, *args):
  return r1(*args) - r2(*args)

def get_reward_diff(r1: RewardCont, r2: RewardCont):
  return partial(_reward_diff, r1, r2)

def canon_callable(
    reward_callable: Callable, env_info, *, n_canon=32, n_norm=96
) -> Callable:
    """
    Return a *vectorised & normalised* canonical reward that accepts
    (s,a,s') NumPy arrays of the same length and returns a float vector.
    """
    can_dict = canon_and_norm_cont(
        reward_callable,
        env_info,
        canon_func_keys=["VAL"],    # just VAL‑2
        norm_opts=[2],
        n_canon_samples=n_canon,
        n_norm_samples=n_norm,
    )
    r = can_dict["VAL-2"]          # grab the one canonical form
    vec_fun = np.vectorize(r)      # vectorise for free
    return vec_fun
