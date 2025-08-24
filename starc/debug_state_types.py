#!/usr/bin/env python3

import numpy as np
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.rewards.ground_truth_reward import GroundTruthReward

def debug_state_types():
    """Debug script to check state types in reward computation"""
    
    print("🔍 Debugging state types in HalfCheetah environment...")
    
    # Create environment
    reward_func = GroundTruthReward()
    env = HalfCheetahEnv(reward_func, discount=0.848, n_episodes_sarsa=100)
    
    # Reset and get initial observation
    obs, _ = env.reset()
    print(f"Initial observation type: {type(obs)}")
    print(f"Initial observation shape: {obs.shape if hasattr(obs, 'shape') else 'No shape'}")
    print(f"Initial observation: {obs}")
    print()
    
    # Take a step
    action = env.action_space.sample()
    print(f"Action type: {type(action)}")
    print(f"Action shape: {action.shape if hasattr(action, 'shape') else 'No shape'}")
    print(f"Action: {action}")
    print()
    
    # Step environment
    next_obs, reward, done, info = env.step(action)
    print(f"Next observation type: {type(next_obs)}")
    print(f"Next observation shape: {next_obs.shape if hasattr(next_obs, 'shape') else 'No shape'}")
    print(f"Next observation: {next_obs}")
    print()
    
    # Test reward function directly
    print("🎯 Testing reward function directly...")
    print(f"env.prev_obs type: {type(env.prev_obs)}")
    print(f"action type: {type(action)}")
    print(f"next_obs type: {type(next_obs)}")
    
    # Test the reward function call
    try:
        test_reward = reward_func(env, env.prev_obs, action, next_obs)
        print(f"Reward from function: {test_reward} (type: {type(test_reward)})")
    except Exception as e:
        print(f"Error calling reward function: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("🔬 Testing vectorized computation...")
    
    # Build small transition batch like in STARC analyzer
    s_list, a_list, sp_list = [], [], []
    obs, _ = env.reset()
    
    for i in range(5):  # Just 5 samples for debugging
        a = env.action_space.sample()
        s = obs.copy()
        obs, *_ = env.step(a)[:4]
        sp = obs.copy()
        
        s_list.append(s)
        a_list.append(a)
        sp_list.append(sp)
        
        print(f"Sample {i}: s type={type(s)}, a type={type(a)}, sp type={type(sp)}")
    
    # Stack into arrays
    S = np.vstack(s_list)
    A = np.vstack(a_list)
    SP = np.vstack(sp_list)
    
    print(f"\nStacked arrays:")
    print(f"S shape: {S.shape}, type: {type(S)}")
    print(f"A shape: {A.shape}, type: {type(A)}")
    print(f"SP shape: {SP.shape}, type: {type(SP)}")
    
    # Test vectorized reward computation
    print("\n🧮 Testing vectorized reward computation...")
    try:
        results = []
        for i, (s, a, sp) in enumerate(zip(S, A, SP)):
            print(f"Transition {i}: s type={type(s)}, a type={type(a)}, sp type={type(sp)}")
            r = reward_func(env, s, a, sp)
            print(f"  -> reward: {r} (type: {type(r)})")
            results.append(r)
        
        vec = np.array(results)
        print(f"\nFinal vector shape: {vec.shape}, type: {type(vec)}")
        print(f"Vector: {vec}")
        
    except Exception as e:
        print(f"Error in vectorized computation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_state_types() 