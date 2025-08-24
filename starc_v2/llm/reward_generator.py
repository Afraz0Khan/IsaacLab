import os
import re
import pathlib
import json
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
import time
from dotenv import load_dotenv

from ..config import STARCv2Config

# Load environment variables
load_dotenv()

class STARCv2RewardGenerator:
    """
    Reward generator for STARC v2 pipeline.
    
    Key differences from v1:
    - Handles LLM-driven reward selection
    - Maintains chat history for context
    - Provides full STARC matrix to LLM for decision making
    """
    
    def __init__(self):
        # Create client with a sane default timeout to avoid hanging indefinitely
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), timeout=300)
        # Detect base directory
        current_dir = pathlib.Path.cwd()
        if current_dir.name == "starc_v2":
            self.base_dir = current_dir
        else:
            self.base_dir = current_dir / "starc_v2"
        
        # Chat history for maintaining context across iterations
        self.chat_history = []
        
        # Prompt templates directory
        self.prompts_dir = self.base_dir / "llm" / "prompts"
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_initial_rewards(self, iteration: int, start_number: int = 1) -> List[pathlib.Path]:
        """Generate initial batch of rewards (iteration 1)"""
        print(f"  🤖 Generating initial {STARCv2Config.REWARDS_PER_ITERATION} reward functions...")
        
        # Load initial prompt (environment-aware) and substitute variables
        initial_prompt = self._load_initial_prompt()
        # Avoid str.format here because the initial prompt contains literal braces
        # in code examples. Do simple token replacement for known placeholders.
        full_prompt = (
            initial_prompt
            .replace("{K}", str(STARCv2Config.REWARDS_PER_ITERATION))
            .replace("{start}", str(start_number))
            .replace("{end}", str(start_number + STARCv2Config.REWARDS_PER_ITERATION - 1))
        )
        
        # Call LLM
        response_text = self._call_llm(full_prompt, is_generation=True)
        if not response_text:
            raise RuntimeError("Failed to generate initial rewards from LLM")
        
        # Parse and save rewards
        # Save under env-specific folder to avoid confusion across environments
        reward_files = self._parse_and_save_rewards(
            response_text,
            f"rewards/{STARCv2Config.ENV_NAME}/iteration_{iteration}"
        )
        
        return reward_files
    
    def generate_rewards_with_selection(self, iteration: int, starc_matrix_str: str, 
                                      start_number: int) -> Tuple[List[str], List[pathlib.Path]]:
        """
        Generate rewards based on LLM selection from STARC matrix.
        
        Returns:
            Tuple of (selected_reward_names, new_reward_files)
        """
        print(f"  🤖 LLM selecting {STARCv2Config.REWARDS_TO_SELECT} rewards and generating {STARCv2Config.REWARDS_PER_ITERATION} new ones...")
        
        # Create selection and generation prompt
        selection_prompt = self._create_selection_prompt(starc_matrix_str, iteration, start_number)
        
        # Call LLM
        response_text = self._call_llm(selection_prompt, is_generation=True)
        if not response_text:
            raise RuntimeError("Failed to get LLM selection and generation")
        
        # Parse selection and new rewards
        selected_rewards, new_reward_files = self._parse_selection_and_rewards(
            response_text,
            f"rewards/{STARCv2Config.ENV_NAME}/iteration_{iteration}"
        )
        
        return selected_rewards, new_reward_files
    
    def _load_initial_prompt(self) -> str:
        """Load the initial prompt for first iteration (env-aware)."""
        env_name = STARCv2Config.ENV_NAME
        filename = {
            'halfcheetah': 'initial_prompt.txt',
            'ant': 'initial_prompt_ant.txt',
            'humanoid': 'initial_prompt_humanoid.txt',
        }.get(env_name, 'initial_prompt.txt')
        prompt_path = self.prompts_dir / filename
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Missing initial prompt template: {prompt_path}")
        
        with open(prompt_path, 'r') as f:
            return f.read()
    
    def _create_selection_prompt(self, starc_matrix_str: str, iteration: int, start_number: int) -> str:
        """Load selection prompt template and inject variables."""
        prompt_path = self.prompts_dir / "selection_prompt.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Missing selection prompt template: {prompt_path}")
        with open(prompt_path, 'r') as f:
            template = f.read()
        return template.format(
            iteration=iteration,
            starc_matrix=starc_matrix_str,
            select_k=STARCv2Config.REWARDS_TO_SELECT,
            K=STARCv2Config.REWARDS_PER_ITERATION,
            start=start_number,
            end=start_number + STARCv2Config.REWARDS_PER_ITERATION - 1,
        )
    
    def _create_default_initial_prompt(self) -> str:
        """Deprecated: initial prompts must be provided in prompts/*.txt"""
        raise RuntimeError("_create_default_initial_prompt is deprecated; provide prompts/*.txt")
    
    def _call_llm(self, prompt: str, is_generation: bool = False) -> Optional[str]:
        """Call OpenAI API with the prompt"""
        try:
            # Allow environment override for model for quick experiments
            model_override = os.getenv('STARC_LLM_MODEL')
            primary_model = model_override or STARCv2Config.PRIMARY_MODEL
            print(f"  📡 Calling OpenAI API ({primary_model}) with 120s timeout...")
            
            # Add to chat history
            self.chat_history.append({"role": "user", "content": prompt})
            
            # Prepare parameters
            # Avoid model-specific token parameter to ensure compatibility across endpoints
            params = {
                "model": primary_model,
                "messages": self.chat_history,
            }
            
            # Add temperature for generation tasks
            if is_generation:
                params["temperature"] = STARCv2Config.TEMPERATURE

            t0 = time.perf_counter()
            response = self.client.chat.completions.create(**params)
            dt = time.perf_counter() - t0
            res = response.choices[0].message.content
            print(f"  ✅ LLM response received in {dt:.1f}s, {len(res or '')} chars")
            
            # Add response to chat history
            self.chat_history.append({"role": "assistant", "content": res})
            
            return res
            
        except Exception as e:
            print(f"  ⚠️  {primary_model} failed: {e}")
            print(f"  📡 Trying fallback ({STARCv2Config.FALLBACK_MODEL}) with 60s timeout...")
            try:
                # Try fallback model
                params["model"] = STARCv2Config.FALLBACK_MODEL
                t1 = time.perf_counter()
                response = self.client.chat.completions.create(**params)
                dt = time.perf_counter() - t1
                res = response.choices[0].message.content
                print(f"  ✅ Fallback response received in {dt:.1f}s, {len(res or '')} chars")
                
                # Add to chat history
                self.chat_history.append({"role": "assistant", "content": res})
                
                return res
            except Exception as e2:
                print(f"  ⚠️  Fallback failed: {e2}")
                # Last resort: try a conservative public model name
                try:
                    backup_model = "o3-mini"
                    print(f"  📡 Trying backup model ({backup_model})...")
                    params["model"] = backup_model
                    t2 = time.perf_counter()
                    response = self.client.chat.completions.create(**params)
                    dt = time.perf_counter() - t2
                    res = response.choices[0].message.content
                    print(f"  ✅ Backup response received in {dt:.1f}s, {len(res or '')} chars")
                    self.chat_history.append({"role": "assistant", "content": res})
                    return res
                except Exception as e3:
                    print(f"  ❌ Backup also failed: {e3}")
                    return None
    
    def _parse_selection_and_rewards(self, response_text: str, output_dir: str) -> Tuple[List[str], List[pathlib.Path]]:
        """Parse LLM response containing both selection and new rewards"""
        # Extract selected rewards
        selected_rewards = []
        selection_match = re.search(r'SELECTED_REWARDS:\s*\n(.*?)\n\n', response_text, re.DOTALL)
        if selection_match:
            selection_text = selection_match.group(1)
            # Extract reward names from bullet points
            for line in selection_text.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('*'):
                    reward_name = line.split('-', 1)[-1].split('*', 1)[-1].strip()
                    if reward_name:
                        selected_rewards.append(reward_name)
        
        # Extract new rewards section
        new_rewards_match = re.search(r'NEW_REWARDS:\s*\n(.*)', response_text, re.DOTALL)
        if new_rewards_match:
            new_rewards_text = new_rewards_match.group(1)
        else:
            # Fallback: use everything after the selection
            new_rewards_text = response_text
        
        # Parse and save new rewards
        new_reward_files = self._parse_and_save_rewards(new_rewards_text, output_dir)
        
        print(f"  ✅ Selected {len(selected_rewards)} rewards: {selected_rewards}")
        
        return selected_rewards, new_reward_files
    
    def _parse_and_save_rewards(self, response_text: str, output_dir: str) -> List[pathlib.Path]:
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

            # Remove stray header line if present
            lines = func_text.splitlines()
            if lines and re.match(r"^RewardFunc_\d+$", lines[0].strip()):
                func_text = "\n".join(lines[1:]).lstrip()

            # Extract class name; if missing, try to auto-wrap into a class
            class_match = re.search(r'class\s+(\w+)', func_text)
            if not class_match:
                # Attempt to detect a top-level def __call__-like body and wrap it
                call_match = re.search(r'def\s+__call__\s*\(', func_text)
                if call_match:
                    class_name = f"RewardFunc_{i+1}"
                    func_text = f"class {class_name}(RewardFunc):\n" + func_text
                else:
                    print(f"    ⚠️  Warning: Could not find class name in function {i+1}")
                    continue

            class_name = class_match.group(1) if class_match else class_name
            
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
    
    def save_chat_history(self, iteration: int):
        """Save chat history for debugging and analysis"""
        history_path = self.base_dir / "results" / f"chat_history_iteration_{iteration}.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(history_path, 'w') as f:
            json.dump(self.chat_history, f, indent=2)
        
        print(f"  💾 Saved chat history to {history_path}")
    
    def load_chat_history(self, filepath: pathlib.Path):
        """Load existing chat history"""
        if filepath.exists():
            with open(filepath, 'r') as f:
                self.chat_history = json.load(f)
            print(f"  📚 Loaded chat history from {filepath}")
        else:
            print(f"  ⚠️  Chat history file not found: {filepath}")
    
    def get_chat_history_summary(self) -> Dict:
        """Get summary of current chat history"""
        return {
            "total_messages": len(self.chat_history),
            "user_messages": len([msg for msg in self.chat_history if msg["role"] == "user"]),
            "assistant_messages": len([msg for msg in self.chat_history if msg["role"] == "assistant"]),
            "total_tokens_estimate": sum(len(msg["content"].split()) for msg in self.chat_history)
        } 