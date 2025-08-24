# STARC v2 - LLM-Driven Reward Evolution Pipeline

STARC v2 is an evolution of the original STARC reward evolution pipeline that removes automated reward selection and instead lets the LLM make all selection decisions based on comprehensive STARC distance information.

## Key Differences from v1

- **LLM-Driven Selection**: No more automated selection of closest/furthest rewards
- **Full STARC Matrix**: LLM receives complete distance matrix including reference bounds
- **Chat History**: OpenAI chat history maintained across iterations for context
- **Reference Bounds**: Includes both ground truth and negative ground truth as reference points
- **Comprehensive Analysis**: More detailed evolution tracking and analysis

## Architecture

```
starc_v2/
├── config.py              # Configuration parameters
├── pipeline.py             # Main pipeline orchestrator
├── core/
│   └── starc_analyzer.py   # STARC distance analysis with references
├── llm/
│   └── reward_generator.py # LLM interaction and reward generation
├── utils/                  # Utility functions
├── rewards/               # Generated reward functions by iteration
└── results/               # Pipeline results and analysis
```

## How It Works

### Iteration 1: Initial Generation
1. LLM generates 16 initial reward functions
2. STARC analysis computes distances including reference bounds
3. Distance matrix saved for next iteration

### Iterations 2+: LLM-Driven Evolution
1. Full STARC matrix (including reference distances) provided to LLM
2. LLM selects 8 rewards to build upon based on matrix analysis
3. LLM generates 16 new reward functions inspired by selections
4. STARC analysis performed on new rewards
5. Process repeats with updated context

## Configuration

Key parameters in `config.py`:

```python
class STARCv2Config:
    REWARDS_PER_ITERATION = 16    # New rewards generated each iteration
    REWARDS_TO_SELECT = 8         # Rewards LLM selects to build upon
    TOTAL_ITERATIONS = 4          # Number of evolution iterations
    N_EPISODES_SARSA = 6000       # SARSA training episodes per reward
    PRIMARY_MODEL = "o1-preview"  # Primary LLM model
    TEMPERATURE = 0.7             # LLM creativity parameter
```

## Usage

```python
from starc_v2.pipeline import STARCv2Pipeline

# Create and run pipeline
pipeline = STARCv2Pipeline()
results = pipeline.run_complete_pipeline()

# Results include:
# - Full evolution history
# - LLM selection decisions
# - Distance matrix evolution
# - Chat history analysis
```

## Output Structure

```
results/
├── final_pipeline_results.json    # Complete pipeline results
├── chat_history_iteration_X.json  # LLM chat history per iteration
└── iteration_X/
    ├── results.json               # Iteration-specific results
    ├── distance_matrix.npy        # Distance matrix (numpy)
    ├── distance_matrix.txt        # Human-readable matrix
    └── reward_names.json          # Reward function names
```

## Key Features

1. **LLM Autonomy**: LLM makes all strategic decisions about reward evolution
2. **Full Context**: Complete distance information including reference bounds
3. **Evolution Tracking**: Detailed analysis of how rewards evolve over iterations
4. **Chat Continuity**: Maintains conversation context across the entire pipeline
5. **Comprehensive Analysis**: Rich statistics and evolution pattern analysis
6. **Fixed GroundTruth Support**: Properly computes x_velocity for expert reward evaluation

## Technical Details

### Environment Fixes
- **HalfCheetahEnvFixed**: Properly computes x_velocity for reward functions that need it
- **Signature Compatibility**: Supports both old and new reward function signatures
- **Ground Truth Support**: GroundTruthReward and NegativeGroundReward work correctly
- **X-Velocity Tracking**: Tracks robot position before/after step to compute velocity

### STARC Distance Matrix
- **Full Matrix**: All pairwise distances between reward functions
- **Reference Bounds**: Distances to both ground truth (expert) and negative ground truth (-expert)
- **Compact Format**: Uses upper triangle only to reduce token usage by ~50%
- **Evolution Context**: Shows how reward landscape changes over iterations

### LLM Integration
- **Model**: Primary o1-preview, fallback to gpt-4o
- **Chat History**: Maintains conversation context across all iterations
- **Selection Strategy**: LLM analyzes full matrix and selects 8 diverse rewards
- **Generation**: LLM creates 16 new reward functions per iteration

## Requirements

- OpenAI API key (set in `.env` file)
- Python 3.8+
- Dependencies listed in `requirements.txt`
- Access to original STARC components (for reward base classes and environment)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
echo "OPENAI_API_KEY=your_key_here" > .env

# Run pipeline
python -m starc_v2.pipeline
```

## Comparison with STARC v1

| Feature | STARC v1 | STARC v2 |
|---------|----------|----------|
| Selection Method | Automated (closest/furthest) | LLM-driven |
| Context | Limited prompt updates | Full chat history |
| Reference Points | Ground truth only | Ground truth + negative ground truth |
| Matrix Information | Truncated summaries | Full distance matrix |
| Decision Making | Rule-based clustering | LLM reasoning |
| Evolution Analysis | Basic metrics | Comprehensive pattern analysis |

STARC v2 represents a shift towards more intelligent, context-aware reward evolution driven by large language model reasoning rather than predefined heuristics. 