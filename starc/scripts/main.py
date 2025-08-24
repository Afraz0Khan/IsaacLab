# Continuous experiment
from multiprocessing import Pool
import numpy as np
import os
import json
from starc.algorithms.distance import canon_and_norm_cont
from starc.algorithms.norm import norm_cont
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.rewards import (
    GroundTruthReward, NegativeGroundReward, PotentialShapedReward,
    RandomReward
)
from starc.utils.utils import get_reward_diff

config = {
  'seed': 42,
  'discount': 0.848,
  'temp_dir': 'temp/continuous',
  # 'canon_func_keys': ['VAL', 'EPIC', 'DARD'],
  'canon_func_keys': ['VAL'],
  # 'norm_opts': [1, 2, float('inf')],
  'norm_opts': [2],
  # 'dist_opts': [1, 2, float('inf')],
  'dist_opts': [2],
  'n_canon_samples': 2**5,
  'n_norm_samples': 96*1,
  'n_episodes_sarsa': 6000,
}

def continuous_experiment(results_path: str):
  np.random.seed(config['seed'])

  # make sure the temp directory exists
  os.makedirs(config['temp_dir'], exist_ok=True)
  # dump the config into it
  with open(f"{config['temp_dir']}/config.json", 'w') as f:
    json.dump(config, f)

  # create the rewards
  ground_truth_env = HalfCheetahEnv(GroundTruthReward(),
                                config['discount'],
                                config['n_episodes_sarsa'])
  non_ground_rews = {
    'potential_shaped': PotentialShapedReward(),
    'negative_ground': NegativeGroundReward(),
    'random': RandomReward(),
  }

  results: dict[str, float] = {}

  # loop over all rewards and compute their distance from the ground truth
  standardized_gt = canon_and_norm_cont(ground_truth_env.reward_func_curried,
                                        ground_truth_env.env_info,
                                        config['canon_func_keys'],
                                        config['norm_opts'],
                                        config['n_canon_samples'],
                                        config['n_norm_samples'])
  for r2_name, r2 in non_ground_rews.items():
    e2 = HalfCheetahEnv(r2, config['discount'], config['n_episodes_sarsa'])
    standardized_r2 = canon_and_norm_cont(e2.reward_func_curried,
                                          e2.env_info,
                                          config['canon_func_keys'],
                                          config['norm_opts'],
                                          config['n_canon_samples'],
                                          config['n_norm_samples'])

    for cn_name in standardized_gt:
      cn_gt = standardized_gt[cn_name]
      cn_r2 = standardized_r2[cn_name]

      for d_ord in config['dist_opts']:
        dist = norm_cont(get_reward_diff(cn_gt, cn_r2),
                         HalfCheetahEnv.predict_next_state,
                         HalfCheetahEnv.state_space,
                         HalfCheetahEnv.act_space,
                         d_ord,
                         config['n_norm_samples'])

        results_key = f'{r2_name}-{cn_name}-{d_ord}'
        print(f'{results_key}: {dist}')
        results[results_key] = dist
        with open(f"{config['temp_dir']}/{results_key}.txt", 'w') as f:
          f.write(str(dist))

  # save as JSON
  with open(results_path, 'w') as f:
    json.dump({'config': config, 'results': results}, f)
  
  # remove temp results directory
  os.system(f"rm -rf {config['temp_dir']}")

if __name__ == '__main__':
  continuous_experiment('results.json')
