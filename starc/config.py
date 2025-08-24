"""
Configuration constants for the Reward Evolution Pipeline

This file contains all configurable parameters for STARC clustering and reward selection.
Adjust these values to fine-tune the pipeline behavior.
"""

class STARCConfig:
    """STARC distance thresholds and clustering parameters"""
    
    # =============================================================================
    # CLUSTERING PARAMETERS
    # =============================================================================
    
    # Minimum STARC distance to form separate clusters
    # Lower values = more clusters (more strict separation)
    # Higher values = fewer clusters (more permissive grouping)
    CLUSTERING_EPS = 0.15
    
    # Minimum number of rewards required to form a cluster
    # Rewards below this threshold become outliers
    CLUSTERING_MIN_SAMPLES = 1
    
    # =============================================================================
    # DIVERSITY SELECTION PARAMETERS
    # =============================================================================
    
    # Minimum STARC distance between cluster representatives
    # Used when selecting 2 representatives from large clusters
    CLUSTER_DIVERSITY_THRESHOLD = 0.1
    
    # Minimum STARC distance between selected rewards in final selection
    # Used to ensure diversity in exploitation/exploration sets
    SELECTION_DIVERSITY_THRESHOLD = 0.05
    
    # =============================================================================
    # REWARD SELECTION PARAMETERS
    # =============================================================================
    
    # Number of closest rewards to select for exploitation
    N_CLOSEST_REWARDS = 4
    
    # Number of furthest rewards to select for exploration  
    N_FURTHEST_REWARDS = 4
    
    # Maximum representatives per cluster (for balanced selection)
    MAX_REPRESENTATIVES_PER_CLUSTER = 2
    
    # =============================================================================
    # PIPELINE PARAMETERS
    # =============================================================================
    
    # Number of reward functions to generate per iteration
    REWARDS_PER_ITERATION = 16
    
    # Total number of pipeline iterations
    TOTAL_ITERATIONS = 3
    
    # =============================================================================
    # LLM PARAMETERS
    # =============================================================================
    
    # Primary model to use for reward generation
    PRIMARY_MODEL = "o3"
    
    # Fallback model if primary fails
    FALLBACK_MODEL = "o3-mini"
    
    @classmethod
    def print_config(cls):
        """Print current configuration values"""
        print("🔧 STARC Pipeline Configuration")
        print("=" * 50)
        print(f"Clustering EPS: {cls.CLUSTERING_EPS}")
        print(f"Cluster diversity threshold: {cls.CLUSTER_DIVERSITY_THRESHOLD}")
        print(f"Selection diversity threshold: {cls.SELECTION_DIVERSITY_THRESHOLD}")
        print(f"Rewards per iteration: {cls.REWARDS_PER_ITERATION}")
        print(f"Total iterations: {cls.TOTAL_ITERATIONS}")
        print(f"Closest/furthest rewards: {cls.N_CLOSEST_REWARDS}/{cls.N_FURTHEST_REWARDS}")
        print("=" * 50)
    
    @classmethod
    def validate_config(cls):
        """Validate configuration values"""
        issues = []
        
        if cls.CLUSTERING_EPS <= 0:
            issues.append("CLUSTERING_EPS must be positive")
            
        if cls.CLUSTER_DIVERSITY_THRESHOLD <= 0:
            issues.append("CLUSTER_DIVERSITY_THRESHOLD must be positive")
            
        if cls.SELECTION_DIVERSITY_THRESHOLD <= 0:
            issues.append("SELECTION_DIVERSITY_THRESHOLD must be positive")
            
        if cls.N_CLOSEST_REWARDS + cls.N_FURTHEST_REWARDS > cls.REWARDS_PER_ITERATION:
            issues.append("Cannot select more rewards than generated per iteration")
            
        if cls.TOTAL_ITERATIONS < 1:
            issues.append("TOTAL_ITERATIONS must be at least 1")
            
        if issues:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {issue}" for issue in issues))
        
        return True


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

class STARCPresets:
    """Preset configurations for different use cases"""
    
    @staticmethod
    def conservative():
        """Conservative settings - high diversity requirements"""
        STARCConfig.CLUSTERING_EPS = 0.20
        STARCConfig.CLUSTER_DIVERSITY_THRESHOLD = 0.15
        STARCConfig.SELECTION_DIVERSITY_THRESHOLD = 0.10
        print("📊 Applied CONSERVATIVE preset (high diversity requirements)")
    
    @staticmethod
    def balanced():
        """Balanced settings - default configuration"""
        STARCConfig.CLUSTERING_EPS = 0.15
        STARCConfig.CLUSTER_DIVERSITY_THRESHOLD = 0.10
        STARCConfig.SELECTION_DIVERSITY_THRESHOLD = 0.05
        print("📊 Applied BALANCED preset (default configuration)")
    
    @staticmethod
    def aggressive():
        """Aggressive settings - allows more similarity"""
        STARCConfig.CLUSTERING_EPS = 0.10
        STARCConfig.CLUSTER_DIVERSITY_THRESHOLD = 0.05
        STARCConfig.SELECTION_DIVERSITY_THRESHOLD = 0.02
        print("📊 Applied AGGRESSIVE preset (allows more similarity)")
    
    @staticmethod
    def experimental():
        """Experimental settings - very strict separation"""
        STARCConfig.CLUSTERING_EPS = 0.25
        STARCConfig.CLUSTER_DIVERSITY_THRESHOLD = 0.20
        STARCConfig.SELECTION_DIVERSITY_THRESHOLD = 0.15
        print("📊 Applied EXPERIMENTAL preset (very strict separation)")


# Validate configuration on import
STARCConfig.validate_config() 