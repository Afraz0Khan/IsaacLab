import os
import re
import pathlib
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
from starc.config import STARCConfig

# Load environment variables
load_dotenv()

class RewardGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        # Detect if we're running from inside starc directory or parent directory
        current_dir = pathlib.Path.cwd()
        if current_dir.name == "starc":
            self.base_dir = current_dir
        else:
            self.base_dir = current_dir / "starc"
        self.message_history = []
        
    def generate_rewards(self, prompt_file: str, output_dir: str, iteration: int, start_number: int = 1) -> List[pathlib.Path]:
        """Generate 16 reward functions using the specified prompt"""
        print(f"  🤖 Using prompt: {prompt_file}")
        
        # Load prompt
        prompt_path = self.base_dir / "llm" / prompt_file
        with open(prompt_path, 'r') as f:
            prompt_content = f.read()
        
        # Add numbering instruction to prompt
        numbering_instruction = f"\n\nIMPORTANT: Number the reward functions starting from RewardFunc_{start_number}. So the first function should be RewardFunc_{start_number}, the second should be RewardFunc_{start_number + 1}, and so on up to RewardFunc_{start_number + STARCConfig.REWARDS_PER_ITERATION - 1}."
        prompt_content += numbering_instruction
        
        # Call LLM
        response_text = self._call_llm(prompt_content)
        if not response_text:
            raise RuntimeError("Failed to generate rewards from LLM")
        
        # Parse and save rewards
        reward_files = self._parse_and_save_rewards(response_text, output_dir, iteration)
        
        # Update __init__.py
        self._update_init_file(output_dir, reward_files)
        
        return reward_files
    
    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI API with the prompt"""
        try:
            print(f"  📡 Calling OpenAI API ({STARCConfig.PRIMARY_MODEL})...")
            self.message_history.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(
                model=STARCConfig.PRIMARY_MODEL,
                messages=self.message_history
            )
            res = response.choices[0].message.content
            self.message_history.append({"role": "assistant", "content": res})
            return res
            
        except Exception as e:
            print(f"  ⚠️  {STARCConfig.PRIMARY_MODEL} failed: {e}")
            print(f"  📡 Trying fallback ({STARCConfig.FALLBACK_MODEL})...")
            try:
                self.message_history.append({"role": "user", "content": prompt})
                response = self.client.chat.completions.create(
                    model=STARCConfig.FALLBACK_MODEL,
                    messages=self.message_history
                )
                res = response.choices[0].message.content
                self.message_history.append({"role": "assistant", "content": res})
                return res
            except Exception as e2:
                print(f"  ❌ Fallback also failed: {e2}")
                return None
    
    def _parse_and_save_rewards(self, response_text: str, output_dir: str, iteration: int) -> List[pathlib.Path]:
        """Parse LLM response and save individual reward functions"""
        # Create output directory
        output_path = self.base_dir / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Split by separator
        reward_functions = response_text.split("--------------------------------")
        
        saved_files = []
        
        for i, func_text in enumerate(reward_functions):
            func_text = func_text.strip()
            if not func_text:
                continue

            # Remove stray header line if present (e.g., "RewardFunc_15")
            lines = func_text.splitlines()
            if lines and re.match(r"^RewardFunc_\\d+$", lines[0].strip()):
                func_text = "\n".join(lines[1:]).lstrip()

            # Extract class name
            class_match = re.search(r'class\s+(\w+)', func_text)
            if not class_match:
                print(f"    ⚠️  Warning: Could not find class name in function {i+1}")
                continue

            class_name = class_match.group(1)
            
            # Clean up code
            func_text = re.sub(r'```python\s*', '', func_text)
            func_text = re.sub(r'```\s*$', '', func_text)
            func_text = func_text.strip()
            
            # Generate filename
            filename = f"{class_name.lower()}.py"
            filepath = output_path / filename
            
            try:
                with open(filepath, 'w') as f:
                    f.write(func_text)
                saved_files.append(filepath)
                print(f"    💾 Saved: {filename}")
                
            except Exception as e:
                print(f"    ❌ Error saving {filename}: {e}")
        
        print(f"  ✅ Saved {len(saved_files)} reward functions")
        return saved_files
    
    def _update_init_file(self, output_dir: str, reward_files: List[pathlib.Path]):
        """Update __init__.py file with imports for new reward functions"""
        init_path = self.base_dir / output_dir / "__init__.py"
        
        imports = []
        for filepath in reward_files:
            module_name = filepath.stem
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    class_match = re.search(r'class\s+(\w+)', content)
                    if class_match:
                        class_name = class_match.group(1)
                        imports.append(f"from .{module_name} import {class_name}")
            except:
                continue
        
        try:
            with open(init_path, 'w') as f:
                f.write("# Auto-generated imports for reward functions\n")
                for imp in imports:
                    f.write(f"{imp}\n")
            print(f"  📝 Updated {init_path} with {len(imports)} imports")
        except Exception as e:
            print(f"  ⚠️  Error updating __init__.py: {e}")
    
    def update_intermediate_prompt(self, closest_rewards: List[str], furthest_rewards: List[str], 
                                 accumulated_pool: List[Dict] = None, iteration: int = 1, global_matrix_str: str = None):
        """Update intermediate prompt with selected reward functions, accumulated pool, and (optionally) global matrix."""
        # Determine which prompt file to create
        if iteration == 1:
            prompt_filename = "intermediate_prompt1.txt"
        else:
            prompt_filename = "intermediate_prompt2.txt"
        prompt_path = self.base_dir / "llm" / prompt_filename
        # Load reward function code
        closest_code = self._load_reward_code(closest_rewards)
        furthest_code = self._load_reward_code(furthest_rewards)
        # Load accumulated pool code if available
        accumulated_code = []
        if accumulated_pool and len(accumulated_pool) > 1:  # Only include if we have previous iterations
            accumulated_code = self._load_accumulated_pool_code(accumulated_pool[:-1])  # Exclude current iteration
        # Create updated prompt
        updated_prompt = self._create_updated_prompt(closest_code, furthest_code, accumulated_code)
        # If a global matrix string is provided (iteration > 1), append it to the prompt
        if global_matrix_str is not None:
            updated_prompt += "\n\n" + global_matrix_str + "\n"
        # Save updated prompt
        with open(prompt_path, 'w') as f:
            f.write(updated_prompt)
        pool_info = f" + {len(accumulated_code)} from accumulated pool" if accumulated_code else ""
        print(f"  📝 Updated {prompt_filename} with {len(closest_rewards)} closest and {len(furthest_rewards)} furthest rewards{pool_info}")
    
    def _load_reward_code(self, reward_names: List[str]) -> List[str]:
        """Load the actual code for specified reward functions"""
        codes = []
        
        # Search in all LLM reward directories
        llm_dir = self.base_dir / "rewards" / "llm"
        
        for reward_name in reward_names:
            found = False
            
            # Search in all iteration directories
            for subdir in llm_dir.iterdir():
                if subdir.is_dir():
                    reward_file = subdir / f"{reward_name}.py"
                    if reward_file.exists():
                        with open(reward_file, 'r') as f:
                            codes.append(f.read())
                        found = True
                        break
            
            if not found:
                print(f"    ⚠️  Warning: Could not find code for {reward_name}")
        
        return codes
    
    def _load_accumulated_pool_code(self, accumulated_pool: List[Dict]) -> List[str]:
        """Load code for all rewards in the accumulated pool from previous iterations"""
        all_reward_names = []
        
        # Collect all reward names from previous iterations
        for iteration_data in accumulated_pool:
            all_reward_names.extend(iteration_data.get('closest', []))
            all_reward_names.extend(iteration_data.get('furthest', []))
        
        # Remove duplicates while preserving order
        unique_names = []
        seen = set()
        for name in all_reward_names:
            if name not in seen:
                unique_names.append(name)
                seen.add(name)
        
        # Load the code
        return self._load_reward_code(unique_names)
    
    def _create_updated_prompt(self, closest_code: List[str], furthest_code: List[str], accumulated_code: List[str] = None) -> str:
        """Create updated intermediate prompt with exploitation and exploration sets"""
        prompt = """Guideline for this round
We already have a seed set of rewards whose mutual STARC distances are ≥ 0.15.
Exploration mode: invent fresh reward functions that are likely to have STARC ≥ 0.15 from every seed (different shaping, new terms, alternate sensors).
Exploitation mode: produce variants that keep the overall intent but tweak weights or smooth out gradients; it's OK if STARC is below 0.15 as long as they plausibly improve learning speed or stability.

+Although the state is sometimes described as a tensor, you can always treat it as a numpy ndarray for all computations.

The first 4 reward functions are the exploitation set (closest to ground truth) while the last 4 are the exploration set (furthest from ground truth).

EXPLOITATION SET (4 closest to ground truth - create variants):
"""
        
        for i, code in enumerate(closest_code[:4], 1):
            prompt += f"\n--- Exploitation Reward {i} ---\n"
            prompt += code + "\n"
        
        prompt += "\nEXPLORATION SET (4 furthest from ground truth - create diverse alternatives):\n"
        
        for i, code in enumerate(furthest_code[:4], 1):
            prompt += f"\n--- Exploration Reward {i} ---\n"
            prompt += code + "\n"
        
        # Add accumulated pool if available
        if accumulated_code:
            prompt += f"\nACCUMULATED POOL ({len(accumulated_code)} high-quality rewards from previous iterations):\n"
            for i, code in enumerate(accumulated_code, 1):
                prompt += f"\n--- Pool Reward {i} ---\n"
                prompt += code + "\n"
        
        prompt += """
Give 4 rewards for exploitation (variants of the closest rewards that might improve performance).
Give 4 rewards for exploration (diverse alternatives inspired by the furthest rewards).
Give 8 more reward functions that do not concern the exploration or exploitation set."""
        
        if accumulated_code:
            prompt += f"""

The accumulated pool shows {len(accumulated_code)} high-quality rewards from previous iterations. You may draw inspiration from these but should not copy them directly. Instead, create novel variations or entirely new approaches."""
        
        prompt += """

You may use all the modules shown in the examples above.
You should return the reward function in a similar format to the examples.
For each reward function you return, name it as "RewardFunc_<number>" and do not use markdown formatting.
Reward functions should be separated by a line of "--------------------------------".
Each reward function should be independent, so make sure to re-import all the modules you need in every reward function.
"""
        
        return prompt 