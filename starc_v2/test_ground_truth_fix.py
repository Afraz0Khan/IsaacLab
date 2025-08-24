#!/usr/bin/env python3
"""
Test script to verify that GroundTruthReward works correctly with the fixed environment
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'starc'))

from starc_v2.core.half_cheetah_env_fixed import HalfCheetahEnvFixed
from starc.rewards.ground_truth_reward import GroundTruthReward
from starc_v2.rewards.negative_ground_reward_fixed import NegativeGroundRewardFixed as NegativeGroundReward

def test_ground_truth_reward():
    """Test that GroundTruthReward works with the fixed environment"""
    print("🧪 Testing GroundTruthReward with fixed environment...")
    
    try:
        # Create environment with GroundTruthReward
        gt_reward = GroundTruthReward()
        env = HalfCheetahEnvFixed(gt_reward, discount=0.848, n_episodes_sarsa=100)
        
        print("✅ Environment created successfully")
        
        # Test reset
        obs = env.reset()
        print(f"✅ Reset successful, observation shape: {obs.shape}")
        
        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            print(f"  Step {i+1}: reward={reward:.4f}, x_velocity={info.get('x_velocity', 'N/A')}")
            
            # Verify reward is a number
            assert isinstance(reward, (int, float)), f"Reward should be a number, got {type(reward)}"
            
        print("✅ GroundTruthReward test passed!")
        return True
        
    except Exception as e:
        print(f"❌ GroundTruthReward test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_negative_ground_reward():
    """Test that NegativeGroundReward works with the fixed environment"""
    print("\n🧪 Testing NegativeGroundReward with fixed environment...")
    
    try:
        # Create environment with NegativeGroundReward
        neg_gt_reward = NegativeGroundReward()
        env = HalfCheetahEnvFixed(neg_gt_reward, discount=0.848, n_episodes_sarsa=100)
        
        print("✅ Environment created successfully")
        
        # Test reset
        obs = env.reset()
        print(f"✅ Reset successful, observation shape: {obs.shape}")
        
        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            print(f"  Step {i+1}: reward={reward:.4f}, x_velocity={info.get('x_velocity', 'N/A')}")
            
            # Verify reward is a number
            assert isinstance(reward, (int, float)), f"Reward should be a number, got {type(reward)}"
            
        print("✅ NegativeGroundReward test passed!")
        return True
        
    except Exception as e:
        print(f"❌ NegativeGroundReward test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_signature_compatibility():
    """Test that the environment works with both old and new reward signatures"""
    print("\n🧪 Testing signature compatibility...")
    
    try:
        # Test with old signature reward function
        class OldSignatureReward:
            def __call__(self, env, prev_obs, action, obs):
                return 1.0  # Simple reward
        
        old_reward = OldSignatureReward()
        env = HalfCheetahEnvFixed(old_reward, discount=0.848, n_episodes_sarsa=100)
        
        obs = env.reset()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        print(f"✅ Old signature reward works: {reward}")
        
        # Test with new signature reward function  
        class NewSignatureReward:
            def __call__(self, env, state, action, next_state, x_velocity):
                return x_velocity  # Return x_velocity as reward
        
        new_reward = NewSignatureReward()
        env2 = HalfCheetahEnvFixed(new_reward, discount=0.848, n_episodes_sarsa=100)
        
        obs = env2.reset()
        action = env2.action_space.sample()
        obs, reward, done, info = env2.step(action)
        
        print(f"✅ New signature reward works: {reward}")
        print("✅ Signature compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Signature compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Testing STARC v2 GroundTruthReward Fix")
    print("=" * 50)
    
    tests = [
        test_ground_truth_reward,
        test_negative_ground_reward,
        test_signature_compatibility
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The GroundTruthReward fix is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 