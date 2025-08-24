#!/usr/bin/env python3
"""
Example usage of STARC configuration system

This script demonstrates how to:
1. Check current configuration
2. Apply preset configurations
3. Manually adjust individual parameters
4. Run the pipeline with custom settings
"""

from starc.config import STARCConfig, STARCPresets
from starc.reward_evolution_pipeline import RewardEvolutionPipeline

def main():
    print("🔧 STARC Configuration Examples")
    print("=" * 50)
    
    # 1. Show current configuration
    print("\n1. Current Configuration:")
    STARCConfig.print_config()
    
    # 2. Apply a preset configuration
    print("\n2. Applying Conservative Preset:")
    STARCPresets.conservative()
    STARCConfig.print_config()
    
    # 3. Manually adjust specific parameters
    print("\n3. Custom Configuration:")
    STARCConfig.CLUSTERING_EPS = 0.12
    STARCConfig.SELECTION_DIVERSITY_THRESHOLD = 0.08
    STARCConfig.N_CLOSEST_REWARDS = 3
    STARCConfig.N_FURTHEST_REWARDS = 5
    STARCConfig.print_config()
    
    # 4. Validate configuration
    try:
        STARCConfig.validate_config()
        print("✅ Configuration is valid!")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return
    
    # 5. Show available presets
    print("\n4. Available Presets:")
    print("  - STARCPresets.conservative() - High diversity requirements")
    print("  - STARCPresets.balanced() - Default balanced settings")  
    print("  - STARCPresets.aggressive() - Allows more similarity")
    print("  - STARCPresets.experimental() - Very strict separation")
    
    # 6. Example of running pipeline with custom config
    print("\n5. Running pipeline with custom configuration...")
    print("   (Uncomment the lines below to actually run)")
    
    # Uncomment to run:
    # pipeline = RewardEvolutionPipeline()
    # results = pipeline.run_complete_pipeline()
    
    # 7. Reset to balanced preset
    print("\n6. Resetting to balanced preset:")
    STARCPresets.balanced()
    STARCConfig.print_config()

if __name__ == "__main__":
    main() 