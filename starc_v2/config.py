"""
Configuration for STARC v2 - LLM-Driven Reward Evolution Pipeline

Key differences from v1:
- No automated closest/furthest selection
- LLM chooses 8 rewards to build upon
- Full STARC matrix provided to LLM including ground truth distances
"""

class STARCv2Config:
    """Configuration for STARC v2 pipeline"""
    
    # =============================================================================
    # PIPELINE PARAMETERS
    # =============================================================================
    
    # Target environment for prompts/pipeline context (does not swap core env yet)
    # Valid options: 'halfcheetah', 'ant', 'humanoid'
    ENV_NAME = "halfcheetah"

    # Number of reward functions to generate per iteration
    REWARDS_PER_ITERATION = 16
    
    # Number of rewards LLM should select to build upon for next iteration
    REWARDS_TO_SELECT = 8
    
    # Total number of pipeline iterations
    TOTAL_ITERATIONS = 3
    
    # =============================================================================
    # SARSA TRAINING PARAMETERS
    # =============================================================================
    
    # Number of episodes for SARSA training per reward
    N_EPISODES_SARSA = 5000
    
    # Discount factor for environment
    DISCOUNT = 0.848
    
    # Number of transition samples for STARC analysis
    N_SAMPLES = 256

    # Parallelism for reward processing (None or 0 → sequential)
    N_WORKERS = 8  # e.g., 4 to process rewards in parallel during analysis
    
    # =============================================================================
    # LLM PARAMETERS
    # =============================================================================
    
    # Primary model for reward generation and selection
    PRIMARY_MODEL = "o3"
    
    # Fallback model if primary fails
    FALLBACK_MODEL = "o3-mini"
    
    # Temperature for LLM calls (0.0 for deterministic, higher for creativity)
    TEMPERATURE = 1.0 # 1.0 is default
    
    # Maximum tokens for LLM responses
    MAX_TOKENS = 4000
    
    # =============================================================================
    # CLUSTERING PARAMETERS
    # =============================================================================
    
    # Clustering configuration (Agglomerative)
    # Use either a distance threshold OR a fixed number of clusters (K).
    # If CLUSTER_N_CLUSTERS is not None, it takes precedence. Otherwise CLUSTER_THRESHOLD is used.
    CLUSTER_THRESHOLD = 0.1  # distance cut (ignored if CLUSTER_N_CLUSTERS is set)
    CLUSTER_N_CLUSTERS = 10  # e.g., 8; set to an int to force exactly K clusters
    
    # =============================================================================
    # MATRIX PRESENTATION PARAMETERS
    # =============================================================================
    
    # Number of decimal places for distance values in matrix
    MATRIX_PRECISION = 5
    
    # Whether to use compact format (upper triangle only) - saves ~50% tokens
    USE_COMPACT_MATRIX = True
    
    # Whether to include full matrix or just upper triangle (deprecated - use USE_COMPACT_MATRIX)
    INCLUDE_FULL_MATRIX = True
    
    # Whether to sort rewards by ground truth distance in matrix presentation
    SORT_BY_GT_DISTANCE = True
    
    @classmethod
    def print_config(cls):
        """Print current configuration values"""
        print("🔧 STARC v2 Pipeline Configuration")
        print("=" * 50)
        print(f"Environment: {cls.ENV_NAME}")
        print(f"Rewards per iteration: {cls.REWARDS_PER_ITERATION}")
        print(f"Rewards to select: {cls.REWARDS_TO_SELECT}")
        print(f"Total iterations: {cls.TOTAL_ITERATIONS}")
        print(f"SARSA episodes: {cls.N_EPISODES_SARSA}")
        print(f"Primary LLM model: {cls.PRIMARY_MODEL}")
        print(f"Temperature: {cls.TEMPERATURE}")
        if cls.CLUSTER_N_CLUSTERS is not None:
            print(f"Clusters (K): {cls.CLUSTER_N_CLUSTERS}")
        else:
            print(f"Cluster threshold: {cls.CLUSTER_THRESHOLD}")
        print("=" * 50)
    
    @classmethod
    def validate_config(cls):
        """Validate configuration values"""
        issues = []
        if cls.ENV_NAME not in {"halfcheetah", "ant", "humanoid"}:
            issues.append("ENV_NAME must be one of {'halfcheetah','ant','humanoid'}")
        
        if cls.REWARDS_TO_SELECT >= cls.REWARDS_PER_ITERATION:
            issues.append("REWARDS_TO_SELECT must be less than REWARDS_PER_ITERATION")
            
        if cls.TOTAL_ITERATIONS < 1:
            issues.append("TOTAL_ITERATIONS must be at least 1")
            
        if cls.N_EPISODES_SARSA < 1:
            issues.append("N_EPISODES_SARSA must be positive")
            
        if cls.TEMPERATURE < 0 or cls.TEMPERATURE > 2:
            issues.append("TEMPERATURE should be between 0 and 2")
            
        if cls.CLUSTER_N_CLUSTERS is None:
            if cls.CLUSTER_THRESHOLD <= 0:
                issues.append("CLUSTER_THRESHOLD must be positive when CLUSTER_N_CLUSTERS is not set")

        if cls.N_WORKERS is not None and cls.N_WORKERS < 0:
            issues.append("N_WORKERS must be None or a non-negative integer")
            
        if issues:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {issue}" for issue in issues))
        
        return True


# Validate configuration on import
STARCv2Config.validate_config() 