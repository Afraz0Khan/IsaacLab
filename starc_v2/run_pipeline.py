#!/usr/bin/env python3
"""
Simple script to run the STARC v2 pipeline
"""

import sys
import os
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import starc_v2
sys.path.append(str(Path(__file__).parent.parent))

from starc_v2.pipeline import STARCv2Pipeline
from starc_v2.config import STARCv2Config

def main():
    """Run the STARC v2 pipeline with CLI overrides (env, etc.)"""
    
    print("🚀 Starting STARC v2 Pipeline")
    print("=" * 50)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["halfcheetah", "ant", "humanoid"], default="halfcheetah",
                        help="Target environment context for prompts and analysis")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Device for SARSA/StateVals (auto, cpu, cuda). Supports multi-GPU via STARC_SARSA_DEVICE env like cuda:0")
    parser.add_argument("--show-usage", action="store_true",
                        help="Print live per-worker progress and CUDA memory usage (if available)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override number of parallel workers for reward analysis")
    parser.add_argument("--n-episodes-sarsa", type=int, default=None,
                        help="Override number of SARSA episodes per reward for canonicalization")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override SARSA training batch size (env: STARC_SARSA_BATCH_SIZE)")
    parser.add_argument("--force-retrain-sarsa", action="store_true",
                        help="Do not load cached SARSA/StateVals checkpoints; always retrain")
    parser.add_argument("--purge-sarsa-cache", action="store_true",
                        help="Delete any existing SARSA/StateVals checkpoints before training")
    args, _ = parser.parse_known_args()

    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your_key_here'")
        print("or create a .env file with OPENAI_API_KEY=your_key_here")
        return 1
    
    try:
        # Apply CLI overrides
        STARCv2Config.ENV_NAME = args.env
        if args.workers is not None:
            STARCv2Config.N_WORKERS = args.workers
        if args.n_episodes_sarsa is not None:
            STARCv2Config.N_EPISODES_SARSA = args.n_episodes_sarsa
        # Device selection for SARSA/StateVals
        if args.device == "cpu":
            os.environ["STARC_SARSA_DEVICE"] = "cpu"
        elif args.device == "cuda":
            os.environ["STARC_SARSA_DEVICE"] = "cuda"
        # Namespace SARSA caches by environment to avoid cross-env reuse
        os.environ["STARC_SARSA_NAMESPACE"] = STARCv2Config.ENV_NAME
        # Force retrain / purge cache controls
        if args.force_retrain_sarsa:
            os.environ["STARC_SARSA_FORCE_RETRAIN"] = "1"
        if args.purge_sarsa_cache:
            os.environ["STARC_SARSA_PURGE_CACHE"] = "1"
        # Optional batch size override
        if args.batch_size is not None and args.batch_size > 0:
            os.environ["STARC_SARSA_BATCH_SIZE"] = str(args.batch_size)
        # Cap intra-op threads to avoid oversubscription under multiprocessing
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        # Enable verbose SARSA progress if requested
        if args.show_usage:
            os.environ["STARC_SARSA_PROGRESS"] = "1"

        # Create and run pipeline
        pipeline = STARCv2Pipeline()
        results = pipeline.run_complete_pipeline()
        
        # Print summary
        print("\n" + "="*70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📊 Total rewards generated: {results['statistics']['total_rewards_generated']}")
        print(f"🧠 Total LLM messages: {results['chat_history_summary']['total_messages']}")
        print(f"⏱️  Total duration: {results['pipeline_duration']}")
        print(f"💾 Results saved to: starc_v2/results/")
        
        # Show evolution summary if available
        if 'evolution_analysis' in results and 'ground_truth_distance_evolution' in results['evolution_analysis']:
            evolution = results['evolution_analysis']['ground_truth_distance_evolution']
            print(f"\n📈 Evolution Summary:")
            print(f"   Initial GT distance (mean): {evolution[0]['mean']:.4f}")
            print(f"   Final GT distance (mean): {evolution[-1]['mean']:.4f}")
            print(f"   Change: {evolution[-1]['mean'] - evolution[0]['mean']:.4f}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 