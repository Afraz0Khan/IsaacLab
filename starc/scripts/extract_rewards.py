#!/usr/bin/env python3
"""
Script to extract all reward functions from starc/rewards/llm and save to a text file.
"""

import pathlib
import datetime

def extract_rewards_to_txt(input_dir="starc/rewards/llm", output_file="all_rewards.txt"):
    """Extract all reward functions to a single text file."""
    
    input_path = pathlib.Path(input_dir)
    reward_files = [f for f in input_path.glob("*.py") if f.name != "__init__.py"]
    
    print(f"🔍 Found {len(reward_files)} reward files in {input_dir}")
    
    with open(output_file, 'w') as outfile:
        # Write header
        outfile.write("=" * 80 + "\n")
        outfile.write("ALL REWARD FUNCTIONS FROM starc/rewards/llm\n")
        outfile.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        outfile.write(f"Total files: {len(reward_files)}\n")
        outfile.write("=" * 80 + "\n\n")
        
        # Process each file
        for i, reward_file in enumerate(sorted(reward_files), 1):
            print(f"  [{i}/{len(reward_files)}] Processing {reward_file.name}")
            
            # Write file separator
            outfile.write(f"\n{'='*60}\n")
            outfile.write(f"FILE {i}: {reward_file.name}\n")
            outfile.write(f"{'='*60}\n\n")
            
            try:
                # Read and write file content
                with open(reward_file, 'r') as infile:
                    content = infile.read()
                    outfile.write(content)
                    
                # Add spacing between files
                outfile.write("\n\n")
                
            except Exception as e:
                outfile.write(f"ERROR reading {reward_file.name}: {e}\n\n")
                print(f"    ❌ Error reading {reward_file.name}: {e}")
                continue
    
    print(f"✅ All reward functions saved to: {output_file}")
    print(f"📄 File size: {pathlib.Path(output_file).stat().st_size} bytes")

if __name__ == "__main__":
    extract_rewards_to_txt() 