# LLM Reward Function Generator

This script uses OpenAI's API to generate 16 reward functions for HalfCheetah based on the prompt in `pilot_prompt.txt`.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your OpenAI API key:
```bash
OPENAI_API_KEY=your_actual_api_key_here
```

## Usage

Run the script:
```bash
python main.py
```

The script will:
1. Load the prompt from `pilot_prompt.txt`
2. Send it to OpenAI API (tries o3-mini first, falls back to gpt-4o)
3. Parse the response and save each reward function as a separate `.py` file in `../starc/rewards/llm/`
4. Create an `__init__.py` file with imports for all generated functions

## Output

Generated reward functions will be saved in:
- `../starc/rewards/llm/rewardfunc_1.py`
- `../starc/rewards/llm/rewardfunc_2.py`
- ... (up to 16 functions)

Each file contains a complete reward function class that can be imported and used in your experiments. 