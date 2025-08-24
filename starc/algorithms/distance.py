from typing import List, Union, Dict
from functools import partial
from starc.algorithms.val import val_canon_cont
from starc.algorithms.norm import norm_cont, norm_cont_with_dbase
from starc.core._types import RewardCont, EnvInfoCont, Space
from starc.utils.simple_utils import timed
import numpy as np
 
canon_funcs = {
    'VAL': val_canon_cont,
}

def _normalized_reward(canonicalized: RewardCont,
                       norm_val: float,
                       s: float,
                       a: float,
                       s_prime: float) -> float:
    if norm_val == 0:
        return canonicalized(s, a, s_prime)
    return canonicalized(s, a, s_prime) / norm_val

# @timed
def canon_and_norm_cont(reward: RewardCont,
                        env_info: EnvInfoCont,
                        canon_func_keys: List[str] = ['VAL'],
                        norm_opts: Union[int, float] = [1, 2, float('inf')],
                        n_canon_samples: int = 10**6,
                        n_norm_samples: int = 10**3) -> Dict[str, RewardCont]:
    """
    Returns a dictionary of all the possible canonicalizations and normalizations
    (lists of possible options are defined in as constants in this file).
    """
    can_r = {c_name: canon_funcs[c_name](reward, env_info, n_canon_samples)
            for c_name in canon_func_keys}
   
    norm_r = {}
    for c_name, val in can_r.items():
        for n_ord in norm_opts:
            norm_val = norm_cont(val,
                                 env_info.trans_dist,
                                 env_info.state_space,
                                 env_info.action_space,
                                 n_ord,
                                 n_norm_samples)
            normalized = partial(_normalized_reward, val, norm_val)
                                                      
            norm_r[f'{c_name}-{n_ord}'] = normalized
    
    return norm_r

# @timed  
def canon_and_norm_cont_with_dbase(reward: RewardCont,
                                   env_info: EnvInfoCont,
                                   states: np.ndarray,
                                   actions: np.ndarray, 
                                   next_states: np.ndarray,
                                   canon_func_keys: List[str] = ['VAL'],
                                   norm_opts: Union[int, float] = [1, 2, float('inf')],
                                   n_canon_samples: int = 10**6,
                                   n_norm_samples: int = 10**3) -> Dict[str, RewardCont]:
    """
    Returns a dictionary of all the possible canonicalizations and normalizations
    using pre-computed transitions database instead of sampling.
    """
    can_r = {c_name: canon_funcs[c_name](reward, env_info, n_canon_samples)
            for c_name in canon_func_keys}
   
    norm_r = {}
    for c_name, val in can_r.items():
        for n_ord in norm_opts:
            norm_val = norm_cont_with_dbase(val,
                                           states, actions, next_states,
                                           n_ord,
                                           n_norm_samples)
            normalized = partial(_normalized_reward, val, norm_val)
                                                      
            norm_r[f'{c_name}-{n_ord}'] = normalized
    
    return norm_r
