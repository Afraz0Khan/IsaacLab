#!/usr/bin/env python3
"""
Reward Evolution Pipeline

This pipeline implements a 4-iteration process:
1. Generate 16 reward functions using pilot prompt
2. Run STARC analysis and clustering
3. Select 4 closest and 4 furthest from ground truth
4. Update intermediate prompt and repeat for 3 more iterations

Each iteration produces:
- 16 new reward functions
- STARC distance analysis
- Clustering results
- Selection of best/worst performers
"""

import os
import json
import shutil
import numpy as np
import pathlib
from typing import List, Dict, Tuple
from datetime import datetime

# Import our modules
from starc.llm.reward_generator import RewardGenerator
from starc.scripts.starc_analyzer import STARCAnalyzer
from starc.scripts.reward_selector import RewardSelector
from starc.utils.pipeline_utils import PipelineUtils
from starc.config import STARCConfig, STARCPresets

class RewardEvolutionPipeline:
    def __init__(self, base_dir: str = None):
        # Auto-detect base directory
        if base_dir is None:
            current_dir = pathlib.Path.cwd()
            if current_dir.name == "starc":
                self.base_dir = current_dir
            else:
                self.base_dir = current_dir / "starc"
        else:
            self.base_dir = pathlib.Path(base_dir)
            
        self.results_dir = self.base_dir / "results" / "pipeline"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.reward_generator = RewardGenerator()
        self.starc_analyzer = STARCAnalyzer()
        self.reward_selector = RewardSelector()
        self.utils = PipelineUtils()
        
        # Pipeline state
        self.iteration = 0
        self.all_results = []
        self.accumulated_rewards_pool = []  # Growing pool of selected rewards
        self.global_reward_counter = 0  # Track reward function numbering across iterations
        self.global_distance_matrix = None  # Will be a numpy array
        self.global_reward_names = []       # List of all reward names in order
        
    def run_complete_pipeline(self) -> Dict:
        """Run the complete 4-iteration pipeline"""
        print("🚀 Starting Reward Evolution Pipeline")
        print("=" * 60)
        
        # Print current configuration
        STARCConfig.print_config()
        
        pipeline_start = datetime.now()
        
        for iteration in range(1, STARCConfig.TOTAL_ITERATIONS + 1):
            print(f"\n🔄 ITERATION {iteration}/{STARCConfig.TOTAL_ITERATIONS}")
            print("-" * 40)
            
            iteration_result = self.run_single_iteration(iteration)
            self.all_results.append(iteration_result)
            
            # Update global distance matrix after each iteration
            self.update_global_distance_matrix(iteration, iteration_result)
            
            # Save intermediate results
            self.save_iteration_results(iteration, iteration_result)
            
        pipeline_end = datetime.now()
        
        # Compile final results
        final_results = {
            "pipeline_duration": str(pipeline_end - pipeline_start),
            "total_iterations": STARCConfig.TOTAL_ITERATIONS,
            "total_rewards_generated": self.global_reward_counter,
            "iterations": self._make_json_safe(self.all_results),
            "accumulated_rewards_pool": self.accumulated_rewards_pool,
            "configuration": {
                "clustering_eps": STARCConfig.CLUSTERING_EPS,
                "cluster_diversity_threshold": STARCConfig.CLUSTER_DIVERSITY_THRESHOLD,
                "selection_diversity_threshold": STARCConfig.SELECTION_DIVERSITY_THRESHOLD,
                "n_closest_rewards": STARCConfig.N_CLOSEST_REWARDS,
                "n_furthest_rewards": STARCConfig.N_FURTHEST_REWARDS,
                "rewards_per_iteration": STARCConfig.REWARDS_PER_ITERATION
            },
            "summary": self.generate_pipeline_summary()
        }
        
        # Save final results
        final_path = self.results_dir / "final_pipeline_results.json"
        with open(final_path, 'w') as f:
            json.dump(final_results, f, indent=2)
            
        print(f"\n🎉 Pipeline Complete!")
        print(f"📊 Final results saved to: {final_path}")
        print(f"⏱️  Total duration: {final_results['pipeline_duration']}")
        
        return final_results
    
    def run_single_iteration(self, iteration: int) -> Dict:
        """Run a single iteration of the pipeline"""
        iteration_start = datetime.now()
        
        print(f"📝 Step 1: Generating {STARCConfig.REWARDS_PER_ITERATION} reward functions...")
        reward_files = self.generate_rewards(iteration)
        
        print(f"🔬 Step 2: Running STARC analysis...")
        starc_results = self.analyze_rewards(iteration, reward_files)
        
        print(f"🎯 Step 3: Clustering rewards...")
        clusters = self.cluster_rewards(iteration, starc_results)
        
        print(f"📊 Step 4: Selecting best/worst performers...")
        selected_rewards = self.select_rewards(iteration, starc_results, clusters)
        
        # Add selected rewards to accumulated pool
        iteration_selections = {
            'iteration': iteration,
            'closest': selected_rewards['closest'],
            'furthest': selected_rewards['furthest'],
            'timestamp': iteration_start.isoformat()
        }
        self.accumulated_rewards_pool.append(iteration_selections)
        
        if iteration < STARCConfig.TOTAL_ITERATIONS:  # Don't update prompt after final iteration
            print(f"📝 Step 5: Updating intermediate prompt...")
            total_pool_size = sum(len(pool['closest']) + len(pool['furthest']) for pool in self.accumulated_rewards_pool)
            print(f"    📚 Accumulated pool now contains {total_pool_size} rewards from {len(self.accumulated_rewards_pool)} iteration(s)")
            self.update_intermediate_prompt(selected_rewards, iteration)
        
        iteration_end = datetime.now()
        
        return {
            "iteration": iteration,
            "duration": str(iteration_end - iteration_start),
            "reward_files": [str(f) for f in reward_files],
            "starc_results": starc_results,
            "clusters": clusters,
            "selected_rewards": selected_rewards,
            "timestamp": iteration_start.isoformat()
        }
    
    def generate_rewards(self, iteration: int, evo_info: Dict = None) -> List[pathlib.Path]:
        """Generate 16 reward functions using LLM"""
        # Determine which prompt to use
        if iteration == 1:
            prompt_file = "pilot_prompt.txt"
        elif iteration == 2:
            prompt_file = "intermediate_prompt1.txt"
        else:
            prompt_file = "intermediate_prompt2.txt"
            
        # Calculate starting number for this iteration
        start_number = self.global_reward_counter + 1
        
        # Generate rewards
        reward_files = self.reward_generator.generate_rewards(
            prompt_file=prompt_file,
            output_dir=f"rewards/llm/iteration_{iteration}",
            iteration=iteration,
            start_number=start_number
        )
        
        # Update global counter
        self.global_reward_counter += len(reward_files)
        
        print(f"  ✅ Generated {len(reward_files)} reward functions (RewardFunc_{start_number} to RewardFunc_{self.global_reward_counter})")
        return reward_files
    
    def analyze_rewards(self, iteration: int, reward_files: List[pathlib.Path]) -> Dict:
        """Run STARC analysis on reward functions"""
        results = self.starc_analyzer.analyze_rewards(
            reward_files=reward_files,
            iteration=iteration
        )
        
        print(f"  ✅ Analyzed {len(reward_files)} rewards")
        print(f"  📏 Distance matrix: {results['distance_matrix'].shape}")
        
        return results
    
    def cluster_rewards(self, iteration: int, starc_results: Dict) -> Dict:
        """Cluster rewards based on STARC distances"""
        clusters = self.starc_analyzer.cluster_rewards(
            distance_matrix=starc_results['distance_matrix'],
            reward_names=starc_results['reward_names'],
            eps=STARCConfig.CLUSTERING_EPS
        )
        
        print(f"  ✅ Found {len(clusters)} clusters")
        for cluster_id, members in clusters.items():
            print(f"    Cluster {cluster_id}: {len(members)} members")
            
        return clusters
    
    def select_rewards(self, iteration: int, starc_results: Dict, clusters: Dict) -> Dict:
        """Select 4 closest and 4 furthest rewards from ground truth using cluster-aware selection"""
        selected = self.reward_selector.select_cluster_aware_best_worst(
            starc_results=starc_results,
            clusters=clusters
        )
        
        print(f"  ✅ Selected rewards:")
        print(f"    🎯 {STARCConfig.N_CLOSEST_REWARDS} closest to ground truth: {selected['closest']}")
        print(f"    🚀 {STARCConfig.N_FURTHEST_REWARDS} furthest from ground truth: {selected['furthest']}")
        
        # Report cluster diversity
        if 'cluster_representatives' in selected:
            cluster_info = selected['cluster_representatives']
            print(f"    📊 From {len(cluster_info)} clusters: {[len(members) for members in cluster_info.values()]} representatives each")
        
        return selected
    
    def update_intermediate_prompt(self, selected_rewards: Dict, iteration: int):
        """Update intermediate prompt with selected rewards, accumulated pool, and (after iter 1) the global matrix."""
        # After first iteration, include the global matrix in the prompt
        global_matrix_str = None
        if iteration > 1 and self.global_distance_matrix is not None:
            global_matrix_str = self.utils.format_distance_matrix_for_prompt(
                self.global_distance_matrix, self.global_reward_names, max_size=32)
        self.reward_generator.update_intermediate_prompt(
            closest_rewards=selected_rewards['closest'],
            furthest_rewards=selected_rewards['furthest'],
            accumulated_pool=self.accumulated_rewards_pool,
            iteration=iteration,
            global_matrix_str=global_matrix_str
        )
        print(f"  ✅ Updated intermediate prompt with current selection + accumulated pool" )
    
    def save_iteration_results(self, iteration: int, results: Dict):
        """Save results for a single iteration"""
        iteration_dir = self.results_dir / f"iteration_{iteration}"
        iteration_dir.mkdir(exist_ok=True)
        
        # Save main results
        with open(iteration_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        # Save distance matrix separately (numpy array)
        if 'starc_results' in results and 'distance_matrix' in results['starc_results']:
            distance_matrix = results['starc_results']['distance_matrix']
            reward_names = results['starc_results']['reward_names']
            
            # Save as numpy binary file
            np.save(
                iteration_dir / "distance_matrix.npy", 
                distance_matrix
            )
            
            # Save as readable text file with headers
            self._save_distance_matrix_txt(
                distance_matrix, 
                reward_names, 
                iteration_dir / "distance_matrix.txt"
            )
            
        print(f"  💾 Saved iteration {iteration} results to {iteration_dir}")
    
    def _make_json_safe(self, data):
        """Recursively remove numpy arrays and other non-JSON-serializable objects"""
        import numpy as np
        
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key == 'distance_matrix' and isinstance(value, np.ndarray):
                    # Skip numpy arrays - they're saved separately
                    result[key + '_shape'] = value.shape  # Keep shape info
                    continue
                elif isinstance(value, np.ndarray):
                    # Convert other numpy arrays to lists
                    result[key] = value.tolist()
                else:
                    result[key] = self._make_json_safe(value)
            return result
        elif isinstance(data, list):
            return [self._make_json_safe(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif hasattr(data, '__dict__'):
            # Handle custom objects by converting to dict
            return self._make_json_safe(data.__dict__)
        else:
            return data
    
    def _save_distance_matrix_txt(self, distance_matrix: np.ndarray, reward_names: List[str], filepath: pathlib.Path):
        """Save distance matrix as a readable text file with headers"""
        try:
            with open(filepath, 'w') as f:
                # Write header
                f.write("STARC Distance Matrix\n")
                f.write("=" * 50 + "\n")
                f.write(f"Matrix size: {distance_matrix.shape[0]} x {distance_matrix.shape[1]}\n")
                f.write(f"Reward functions: {len(reward_names)}\n\n")
                
                # Write column headers
                f.write("Reward Names:\n")
                for i, name in enumerate(reward_names):
                    f.write(f"{i:2d}: {name}\n")
                f.write("\n")
                
                # Write matrix with row/column indices
                f.write("Distance Matrix (rows=from, cols=to):\n")
                f.write("     ")
                for i in range(len(reward_names)):
                    f.write(f"{i:8d}")
                f.write("\n")
                
                for i, row in enumerate(distance_matrix):
                    f.write(f"{i:2d}: ")
                    for j, value in enumerate(row):
                        f.write(f"{value:8.4f}")
                    f.write(f"  # {reward_names[i]}\n")
                
                # Write summary statistics
                f.write(f"\nSummary Statistics:\n")
                f.write(f"Mean distance: {np.mean(distance_matrix):.4f}\n")
                f.write(f"Std distance:  {np.std(distance_matrix):.4f}\n")
                f.write(f"Min distance:  {np.min(distance_matrix):.4f}\n")
                f.write(f"Max distance:  {np.max(distance_matrix):.4f}\n")
                
                # Write pairwise distances (excluding diagonal)
                upper_triangle = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
                f.write(f"Mean pairwise distance: {np.mean(upper_triangle):.4f}\n")
                
            print(f"    📄 Saved readable distance matrix to {filepath.name}")
            
        except Exception as e:
            print(f"    ⚠️  Warning: Could not save distance matrix text file: {e}")
    
    def generate_pipeline_summary(self) -> Dict:
        """Generate a summary of the entire pipeline"""
        summary = {
            "total_rewards_generated": self.global_reward_counter,
            "iterations_completed": len(self.all_results),
            "final_clusters": self.all_results[-1]['clusters'] if self.all_results else {},
            "rewards_per_iteration": [len(r['reward_files']) for r in self.all_results],
            "selected_rewards_per_iteration": [
                len(r['selected_rewards']['closest']) + len(r['selected_rewards']['furthest']) 
                for r in self.all_results if 'selected_rewards' in r
            ],
            "evolution_metrics": self.calculate_evolution_metrics()
        }
        
        return summary
    
    def calculate_evolution_metrics(self) -> Dict:
        """Calculate metrics showing how rewards evolved over iterations"""
        if len(self.all_results) < 2:
            return {}
            
        metrics = {
            "diversity_trend": [],
            "performance_trend": [],
            "cluster_count_trend": []
        }
        
        for result in self.all_results:
            # Calculate diversity (average pairwise distance)
            if 'starc_results' in result and 'distance_matrix' in result['starc_results']:
                dist_matrix = result['starc_results']['distance_matrix']
                avg_distance = np.mean(dist_matrix[np.triu_indices_from(dist_matrix, k=1)])
                metrics['diversity_trend'].append(float(avg_distance))
            
            # Count clusters
            if 'clusters' in result:
                metrics['cluster_count_trend'].append(len(result['clusters']))
        
        return metrics

    def update_global_distance_matrix(self, iteration: int, iteration_result: Dict):
        """
        Update the global STARC distance matrix and reward name list after each iteration.
        """
        starc_results = iteration_result['starc_results']
        new_names = starc_results['reward_names']
        new_vectors = starc_results['vectors']
        # On first iteration, just use the current matrix
        if self.global_distance_matrix is None:
            self.global_distance_matrix = starc_results['distance_matrix'].copy()
            self.global_reward_names = list(new_names)
        else:
            # Expand the matrix to include new rewards
            old_n = len(self.global_reward_names)
            new_n = len(new_names)
            total_n = old_n + new_n
            new_matrix = np.zeros((total_n, total_n), dtype=np.float32)
            # Copy old block
            new_matrix[:old_n, :old_n] = self.global_distance_matrix
            # Compute distances between old and new
            for i, old_vec in enumerate(self._get_all_vectors()):
                for j, new_vec in enumerate(new_vectors):
                    dist = np.linalg.norm(np.array(old_vec) - np.array(new_vec))
                    new_matrix[i, old_n + j] = dist
                    new_matrix[old_n + j, i] = dist
            # Fill new block (new rewards among themselves)
            for i in range(new_n):
                for j in range(new_n):
                    if i != j:
                        dist = np.linalg.norm(np.array(new_vectors[i]) - np.array(new_vectors[j]))
                        new_matrix[old_n + i, old_n + j] = dist
            self.global_distance_matrix = new_matrix
            self.global_reward_names.extend(new_names)
        # Save global matrix to disk
        global_dir = self.results_dir
        np.save(global_dir / "global_distance_matrix.npy", self.global_distance_matrix)
        with open(global_dir / "global_reward_names.json", 'w') as f:
            json.dump(self.global_reward_names, f, indent=2)
        # Also save as readable text
        matrix_txt = self.utils.format_distance_matrix_for_prompt(self.global_distance_matrix, self.global_reward_names, max_size=32)
        with open(global_dir / "global_distance_matrix.txt", 'w') as f:
            f.write(matrix_txt)

    def _get_all_vectors(self):
        """Helper to get all reward vectors in order so far (for global matrix update)."""
        vectors = []
        for result in self.all_results:
            if 'starc_results' in result and 'vectors' in result['starc_results']:
                vectors.extend(result['starc_results']['vectors'])
        return vectors

def main():
    """Main entry point for the pipeline"""
    print("🎯 Reward Evolution Pipeline")
    print(f"Generating and evolving reward functions over {STARCConfig.TOTAL_ITERATIONS} iterations")
    print()
    
    # Show configuration options
    print("💡 Configuration Options:")
    print("   To adjust STARC thresholds, edit starc/config.py or use presets:")
    print("   - STARCPresets.conservative() - High diversity requirements")
    print("   - STARCPresets.balanced() - Default settings")
    print("   - STARCPresets.aggressive() - Allow more similarity")
    print("   - STARCPresets.experimental() - Very strict separation")
    print()
    
    # Check environment
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file")
        return
    
    # Initialize and run pipeline
    pipeline = RewardEvolutionPipeline()
    results = pipeline.run_complete_pipeline()
    
    # Print final summary
    print("\n📈 PIPELINE SUMMARY")
    print("=" * 50)
    summary = results['summary']
    print(f"Total rewards generated: {summary['total_rewards_generated']} (RewardFunc_1 to RewardFunc_{summary['total_rewards_generated']})")
    print(f"Iterations completed: {summary['iterations_completed']}")
    print(f"Rewards per iteration: {summary['rewards_per_iteration']}")
    print(f"Selected per iteration: {summary['selected_rewards_per_iteration']}")
    print(f"Final cluster count: {len(summary['final_clusters'])}")
    
    if 'evolution_metrics' in summary and summary['evolution_metrics']:
        metrics = summary['evolution_metrics']
        if 'diversity_trend' in metrics and metrics['diversity_trend']:
            print(f"Diversity evolution: {metrics['diversity_trend'][0]:.3f} → {metrics['diversity_trend'][-1]:.3f}")
        if 'cluster_count_trend' in metrics and metrics['cluster_count_trend']:
            print(f"Cluster count evolution: {metrics['cluster_count_trend'][0]} → {metrics['cluster_count_trend'][-1]}")

if __name__ == "__main__":
    main() 