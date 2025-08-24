# STARC: Structured Analysis of Reward Clustering

A comprehensive framework for analyzing and clustering reward functions in reinforcement learning environments.

## 📁 Project Structure

```
starc/
├── 📁 core/                    # Core framework components
│   ├── half_cheetah_env.py     # Custom HalfCheetah environment
│   ├── reward_func.py          # Base reward function class
│   ├── state_vals.py           # State value computation
│   └── _types.py               # Type definitions
│
├── 📁 algorithms/              # Machine learning algorithms
│   ├── sarsa.py                # SARSA implementation
│   ├── val.py                  # Value-based canonicalization
│   ├── distance.py             # Distance metrics
│   └── norm.py                 # Normalization functions
│
├── 📁 utils/                   # Utility functions
│   ├── simple_utils.py         # Basic utilities (timed, sample_space)
│   ├── utils.py                # Advanced utilities
│   └── sample_reward_worker.py # Reward sampling utilities
│
├── 📁 scripts/                 # Analysis and execution scripts
│   ├── main.py                 # Main comparison script
│   ├── cluster_rewards.py      # Reward clustering analysis
│   ├── extract_rewards.py      # Extract rewards to text
│   └── build_dbase.py          # Database building
│
├── 📁 llm/                     # LLM reward generation
│   ├── main.py                 # Generate rewards using OpenAI
│   ├── pilot_prompt.txt        # Main generation prompt
│   ├── intermediate_prompt.txt # Intermediate prompt
│   ├── starc_info.txt          # STARC framework info
│   └── requirements.txt        # LLM-specific requirements
│
├── 📁 rewards/                 # Reward function implementations
│   ├── ground_truth_reward.py  # Original HalfCheetah reward
│   ├── negative_ground_reward.py
│   ├── potential_shaped_reward.py
│   ├── random_reward.py
│   └── llm/                    # LLM-generated rewards
│       ├── rewardfunc_1.py
│       ├── rewardfunc_2.py
│       └── ... (16 total)
│
├── 📁 results/                 # Output files and results
│   ├── clusters.json           # Clustering results
│   ├── all_rewards.txt         # Extracted reward functions
│   └── results*.json           # Experiment results
│
├── 📁 data/                    # Data files
├── 📁 sarsa_models/            # Trained SARSA models
└── 📁 temp/                    # Temporary files
```

## 🚀 Quick Start

### 1. Generate New Reward Functions
```bash
cd starc/llm
conda activate openai3.9
python main.py
```

### 2. Run Reward Clustering
```bash
cd /path/to/openai
conda activate openai3.9
python starc/scripts/cluster_rewards.py
```

### 3. Compare Rewards
```bash
cd /path/to/openai
conda activate openai3.9
python starc/scripts/main.py
```

### 4. Extract All Rewards to Text
```bash
cd /path/to/openai
python starc/scripts/extract_rewards.py
```

## 📊 Recent Results

The latest clustering analysis (in `results/clusters.json`) identified 9 distinct clusters:

- **Cluster 0** (6 functions): Similar reward patterns
- **Cluster 1** (3 functions): Alternative approach group
- **Clusters 2-8**: Individual unique reward functions

## 🔧 Key Features

- **Modular Design**: Clean separation of concerns
- **LLM Integration**: Generate new reward functions using OpenAI
- **Clustering Analysis**: Identify similar reward behaviors
- **SARSA Training**: State value computation for canonicalization
- **Comprehensive Testing**: Multiple reward function comparisons

## 📝 Usage Examples

```python
# Import core components
from starc.core import HalfCheetahEnv, RewardFunc
from starc.rewards import GroundTruthReward
from starc.algorithms import train_sarsa_model

# Create environment
env = HalfCheetahEnv(GroundTruthReward(), discount=0.848, n_episodes_sarsa=1000)

# Generate new rewards using LLM
from starc.llm import generate_reward_functions
new_rewards = generate_reward_functions()
```

## 🛠️ Requirements

- Python 3.9+
- PyTorch
- NumPy
- scikit-learn
- OpenAI API (for LLM generation)
- Custom gym environments

## 📈 Performance

- **Clustering**: ~1.5 hours for 16 reward functions (1000 episodes each)
- **SARSA Training**: ~18 minutes per reward function
- **Memory Usage**: Optimized for standard workstation hardware