#!/usr/bin/env python3
"""
Reset script for STARC reward evolution pipeline.

This script cleans up all iteration-related files and models:
1. Deletes all iteration reward functions in rewards/llm/
2. Deletes the pipeline folder in results/
3. Deletes all associated SARSA models (RewardFunc_* numbered models)
4. Resets intermediate prompts to their original state
5. Preserves core reward functions (GroundTruthReward, RandomReward, etc.)
"""

import os
import shutil
import glob
import re
from pathlib import Path

def get_script_dir():
    """Get the directory where this script is located (starc/)"""
    return Path(__file__).parent.absolute()

def reset_intermediate_prompts():
    """Reset intermediate prompts to their original state (basic templates)"""
    script_dir = get_script_dir()
    llm_dir = script_dir / "llm"
    
    if not llm_dir.exists():
        print(f"LLM directory {llm_dir} does not exist")
        return
    
    print(f"Resetting intermediate prompts in {llm_dir}")
    
    # Original intermediate prompt template (basic template without rewards)
    original_intermediate_template = """Guideline for this round
We already have a seed set of rewards whose mutual STARC distances are ≥ 0.15.
Exploration mode: invent fresh reward functions that are likely to have STARC ≥ 0.15 from every seed (different shaping, new terms, alternate sensors).
Exploitation mode: produce variants that keep the overall intent but tweak weights or smooth out gradients; it's OK if STARC is below 0.15 as long as they plausibly improve learning speed or stability.

The first 4 reward functions are the exploitation set (closest to ground truth) while the last 4 are the exploration set (furthest from ground truth).

EXPLOITATION SET (4 closest to ground truth - create variants):

[To be populated with closest rewards from STARC analysis]

EXPLORATION SET (4 furthest from ground truth - create diverse alternatives):

[To be populated with furthest rewards from STARC analysis]

Give 4 rewards for exploitation (variants of the closest rewards that might improve performance).
Give 4 rewards for exploration (diverse alternatives inspired by the furthest rewards).
Give 8 more reward functions that do not concern the exploration or exploitation set.

You may use all the modules shown in the examples above.
You should return the reward function in a similar format to the examples.
For each reward function you return, name it as "RewardFunc_<number>" and do not use markdown formatting.
Reward functions should be separated by a line of "--------------------------------".
Each reward function should be independent, so make sure to re-import all the modules you need in every reward function.
"""
    
    # Reset intermediate prompt files
    prompt_files = ["intermediate_prompt1.txt", "intermediate_prompt2.txt"]
    reset_count = 0
    
    for prompt_file in prompt_files:
        prompt_path = llm_dir / prompt_file
        if prompt_path.exists():
            print(f"  Resetting {prompt_file}")
            with open(prompt_path, 'w') as f:
                f.write(original_intermediate_template)
            reset_count += 1
        else:
            print(f"  Creating {prompt_file}")
            with open(prompt_path, 'w') as f:
                f.write(original_intermediate_template)
            reset_count += 1
    
    print(f"Reset {reset_count} intermediate prompt files")

def delete_iteration_rewards():
    """Delete all iteration reward functions in rewards/llm/"""
    script_dir = get_script_dir()
    llm_rewards_dir = script_dir / "rewards" / "llm"
    
    if not llm_rewards_dir.exists():
        print(f"Directory {llm_rewards_dir} does not exist")
        return
    
    print(f"Cleaning up iteration rewards in {llm_rewards_dir}")
    
    # Delete iteration directories
    iteration_dirs = list(llm_rewards_dir.glob("iteration_*"))
    deleted_count = 0
    
    for iteration_dir in iteration_dirs:
        if iteration_dir.is_dir():
            print(f"  Deleting {iteration_dir.name}/")
            shutil.rmtree(iteration_dir)
            deleted_count += 1
    
    # Clean up __pycache__ if it exists
    pycache_dir = llm_rewards_dir / "__pycache__"
    if pycache_dir.exists():
        print(f"  Deleting __pycache__/")
        shutil.rmtree(pycache_dir)
    
    print(f"Deleted {deleted_count} iteration directories from rewards/llm/")

def delete_pipeline_results():
    """Delete the pipeline folder in results/"""
    script_dir = get_script_dir()
    pipeline_dir = script_dir / "results" / "pipeline"
    
    if not pipeline_dir.exists():
        print(f"Pipeline directory {pipeline_dir} does not exist")
        return
    
    print(f"Deleting pipeline results directory: {pipeline_dir}")
    shutil.rmtree(pipeline_dir)
    print("Pipeline results directory deleted")

def delete_iteration_sarsa_models():
    """Delete SARSA models for iteration reward functions (RewardFunc_* numbered models)"""
    script_dir = get_script_dir()
    sarsa_models_dir = script_dir / "sarsa_models"
    
    if not sarsa_models_dir.exists():
        print(f"SARSA models directory {sarsa_models_dir} does not exist")
        return
    
    print(f"Cleaning up iteration SARSA models in {sarsa_models_dir}")
    
    # Pattern to match RewardFunc_<number> directories
    # These correspond to the numbered reward functions generated by iterations
    reward_func_dirs = list(sarsa_models_dir.glob("RewardFunc_*"))
    
    # Core reward functions to preserve (not numbered iteration rewards)
    core_rewards = {
        "GroundTruthReward",
        "RandomReward", 
        "NegativeGroundReward",
        "PotentialShapedReward",
        "function"  # Keep this as well
    }
    
    deleted_count = 0
    
    for model_dir in reward_func_dirs:
        if model_dir.is_dir():
            dir_name = model_dir.name
            
            # Check if this is a numbered reward function (RewardFunc_<number>)
            if re.match(r'^RewardFunc_\d+$', dir_name):
                print(f"  Deleting {dir_name}/")
                shutil.rmtree(model_dir)
                deleted_count += 1
            else:
                print(f"  Preserving {dir_name}/ (core reward)")
    
    # Also preserve the explicitly named core reward directories
    for core_dir in sarsa_models_dir.iterdir():
        if core_dir.is_dir() and core_dir.name in core_rewards:
            print(f"  Preserving {core_dir.name}/ (core reward)")
    
    print(f"Deleted {deleted_count} iteration SARSA model directories")



def main():
    """Main reset function"""
    print("="*60)
    print("STARC REWARD EVOLUTION PIPELINE RESET")
    print("="*60)
    print()
    
    script_dir = get_script_dir()
    print(f"Working directory: {script_dir}")
    print()
    
    # Confirm with user before proceeding
    response = input("This will delete all iteration rewards, pipeline results, SARSA models, and reset prompts. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Reset cancelled.")
        return
    
    print("\nStarting reset process...\n")
    
    # Step 1: Delete iteration rewards
    print("Step 1: Deleting iteration reward functions")
    delete_iteration_rewards()
    print()
    
    # Step 2: Delete pipeline results
    print("Step 2: Deleting pipeline results")
    delete_pipeline_results()
    print()
    
    # Step 3: Delete iteration SARSA models
    print("Step 3: Deleting iteration SARSA models")
    delete_iteration_sarsa_models()
    print()
    
    # Step 4: Reset intermediate prompts
    print("Step 4: Resetting intermediate prompts")
    reset_intermediate_prompts()
    print()
    
    print("="*60)
    print("RESET COMPLETE")
    print("="*60)
    print()
    print("The following have been preserved:")
    print("- Core reward functions (GroundTruthReward, RandomReward, etc.)")
    print("- Core SARSA models for base rewards")
    print("- Pilot prompt file (pilot_prompt.txt)")
    print("- General results files (clusters.json, etc.)")
    print()
    print("The following have been reset:")
    print("- Intermediate prompt files (reset to template state)")
    print("- All iteration-specific rewards and results")
    print()
    print("Ready for new pipeline execution!")

if __name__ == "__main__":
    main() 