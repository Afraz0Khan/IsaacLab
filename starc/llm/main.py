import os
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_pilot_prompt():
    """Load the pilot prompt from file"""
    with open('pilot_prompt.txt', 'r') as f:
        return f.read()

def generate_reward_functions():
    """Generate 16 reward functions using OpenAI API"""
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    pilot_prompt = load_pilot_prompt()
    
    print("Generating 16 reward functions using o3-mini...")
    
    try:
        response = client.chat.completions.create(
            model="o3", 
            messages=[
                {
                    "role": "user", 
                    "content": pilot_prompt
                }
            ],
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        print("Trying fallback model o3-mini...")
        try:
            response = client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {
                        "role": "user", 
                        "content": pilot_prompt
                    }
                ],
            )
            return response.choices[0].message.content
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return None

def parse_and_save_reward_functions(response_text):
    """Parse the response and save each reward function as a separate file"""
    
    # Create the output directory
    output_dir = "../rewards/llm"
    os.makedirs(output_dir, exist_ok=True)
    
    # Split by the separator
    reward_functions = response_text.split("--------------------------------")
    
    saved_count = 0
    
    for i, func_text in enumerate(reward_functions):
        func_text = func_text.strip()
        if not func_text:
            continue
            
        # Extract class name using regex
        class_match = re.search(r'class\s+(\w+)', func_text)
        if not class_match:
            print(f"Warning: Could not find class name in function {i+1}")
            continue
            
        class_name = class_match.group(1)
        
        # Clean up the code - remove markdown formatting if present
        func_text = re.sub(r'```python\s*', '', func_text)
        func_text = re.sub(r'```\s*$', '', func_text)
        func_text = func_text.strip()
        
        # Save to file
        filename = f"{class_name.lower()}.py"
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                f.write(func_text)
            print(f"Saved: {filepath}")
            saved_count += 1
            
        except Exception as e:
            print(f"Error saving {filename}: {e}")
    
    print(f"\nSuccessfully saved {saved_count} reward functions to {output_dir}")
    return saved_count

def update_init_file(saved_count):
    """Update the __init__.py file to include the new reward functions"""
    init_file_path = "../rewards/llm/__init__.py"
    
    # Get list of generated files
    llm_dir = "../rewards/llm"
    if not os.path.exists(llm_dir):
        return
        
    py_files = [f for f in os.listdir(llm_dir) if f.endswith('.py') and f != '__init__.py']
    
    # Generate import statements
    imports = []
    for py_file in py_files:
        module_name = py_file[:-3]  # Remove .py extension
        # Try to extract class name from file
        try:
            with open(os.path.join(llm_dir, py_file), 'r') as f:
                content = f.read()
                class_match = re.search(r'class\s+(\w+)', content)
                if class_match:
                    class_name = class_match.group(1)
                    imports.append(f"from .{module_name} import {class_name}")
        except:
            continue
    
    # Write __init__.py
    try:
        with open(init_file_path, 'w') as f:
            f.write("# Auto-generated imports for LLM reward functions\n")
            for imp in imports:
                f.write(f"{imp}\n")
        print(f"Updated {init_file_path} with {len(imports)} imports")
    except Exception as e:
        print(f"Error updating __init__.py: {e}")

def main():
    print("Starting reward function generation...")
    
    # Check if API key exists
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return
    
    # Generate reward functions
    response = generate_reward_functions()
    if not response:
        print("Failed to generate reward functions")
        return
    
    print("Response received. Parsing and saving...")
    
    # Parse and save
    saved_count = parse_and_save_reward_functions(response)
    
    if saved_count > 0:
        # Update __init__.py
        update_init_file(saved_count)
        print(f"\nGeneration complete! {saved_count} reward functions created.")
    else:
        print("No reward functions were saved successfully.")

if __name__ == "__main__":
    main() 