#!/usr/bin/env python3
"""
Script to fix import statements after reorganizing the starc folder.
"""

import os
import re
from pathlib import Path

def fix_imports():
    """Fix import statements in all Python files after reorganization."""
    
    # Define the import mappings
    import_fixes = {
        # Core imports
        r'from half_cheetah_env import': 'from starc.core.half_cheetah_env import',
        r'from reward_func import': 'from starc.core.reward_func import',
        r'from state_vals import': 'from starc.core.state_vals import',
        r'from _types import': 'from starc.core._types import',
        
        # Algorithm imports
        r'from sarsa import': 'from starc.algorithms.sarsa import',
        r'from val import': 'from starc.algorithms.val import',
        r'from distance import': 'from starc.algorithms.distance import',
        r'from norm import': 'from starc.algorithms.norm import',
        
        # Utils imports
        r'from simple_utils import': 'from starc.utils.simple_utils import',
        r'from utils import': 'from starc.utils.utils import',
    }
    
    # Files to process
    files_to_fix = [
        'core/state_vals.py',
        'algorithms/sarsa.py',
        'algorithms/val.py',
        'algorithms/distance.py',
        'algorithms/norm.py',
        'utils/utils.py',
        'scripts/main.py',
        'scripts/cluster_rewards.py',
        'scripts/extract_rewards.py',
        'rewards/ground_truth_reward.py',
        'rewards/negative_ground_reward.py',
        'rewards/potential_shaped_reward.py',
        'rewards/random_reward.py',
    ]
    
    for file_path in files_to_fix:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"Fixing imports in {file_path}")
            
            with open(full_path, 'r') as f:
                content = f.read()
            
            # Apply fixes
            for old_pattern, new_import in import_fixes.items():
                content = re.sub(old_pattern, new_import, content)
            
            with open(full_path, 'w') as f:
                f.write(content)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    fix_imports()
    print("✅ Import fixes completed!") 