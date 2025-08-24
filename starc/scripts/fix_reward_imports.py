#!/usr/bin/env python3
"""
Script to fix import statements in all reward function files.
"""

import os
import re
from pathlib import Path

def fix_reward_imports():
    """Fix import statements in all reward function files."""
    
    # Define the import mappings
    import_fixes = {
        r'from core\.half_cheetah_env import': 'from starc.core.half_cheetah_env import',
        r'from core\.reward_func import': 'from starc.core.reward_func import',
        r'from half_cheetah_env import': 'from starc.core.half_cheetah_env import',
        r'from reward_func import': 'from starc.core.reward_func import',
    }
    
    # Get all reward function files from both directories
    rewards_llm_dir = Path("../rewards/llm")
    rewards_main_dir = Path("../rewards")
    
    reward_files = []
    reward_files.extend(list(rewards_llm_dir.glob("rewardfunc_*.py")))
    reward_files.extend(list(rewards_main_dir.glob("*_reward.py")))
    
    print(f"Found {len(reward_files)} reward function files")
    
    for reward_file in reward_files:
        print(f"Fixing imports in {reward_file.name}")
        
        # Read file content
        with open(reward_file, 'r') as f:
            content = f.read()
        
        # Apply fixes
        original_content = content
        for old_pattern, new_import in import_fixes.items():
            content = re.sub(old_pattern, new_import, content)
        
        # Write back if changed
        if content != original_content:
            with open(reward_file, 'w') as f:
                f.write(content)
            print(f"  ✅ Updated {reward_file.name}")
        else:
            print(f"  ⏭️  No changes needed for {reward_file.name}")

if __name__ == "__main__":
    fix_reward_imports()
    print("✅ Reward function import fixes completed!") 