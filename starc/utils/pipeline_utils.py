import os
import json
import shutil
import pathlib
import numpy as np
from typing import Dict, List, Any
from datetime import datetime

class PipelineUtils:
    def __init__(self):
        pass
    
    def create_directory_structure(self, base_dir: str) -> Dict[str, pathlib.Path]:
        """Create the necessary directory structure for the pipeline"""
        base_path = pathlib.Path(base_dir)
        
        directories = {
            'results': base_path / "results" / "pipeline",
            'rewards_base': base_path / "rewards" / "llm",
            'temp': base_path / "temp" / "pipeline"
        }
        
        for name, path in directories.items():
            path.mkdir(parents=True, exist_ok=True)
        
        return directories
    
    def backup_existing_rewards(self, rewards_dir: pathlib.Path) -> pathlib.Path:
        """Backup existing reward functions before starting pipeline"""
        if not rewards_dir.exists():
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = rewards_dir.parent / f"llm_backup_{timestamp}"
        
        if rewards_dir.exists() and any(rewards_dir.iterdir()):
            shutil.copytree(rewards_dir, backup_dir)
            print(f"  💾 Backed up existing rewards to: {backup_dir}")
            return backup_dir
        
        return None
    
    def save_json_with_numpy(self, data: Dict, filepath: pathlib.Path):
        """Save JSON data with numpy array handling"""
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        # Convert numpy objects
        converted_data = self._deep_convert(data, convert_numpy)
        
        with open(filepath, 'w') as f:
            json.dump(converted_data, f, indent=2, default=str)
    
    def _deep_convert(self, obj: Any, converter) -> Any:
        """Recursively convert objects in nested structures"""
        if isinstance(obj, dict):
            return {key: self._deep_convert(value, converter) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_convert(item, converter) for item in obj]
        else:
            return converter(obj)
    
    def load_json_with_numpy(self, filepath: pathlib.Path) -> Dict:
        """Load JSON data and convert lists back to numpy arrays where appropriate"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert specific fields back to numpy arrays
        if 'distance_matrix' in data and isinstance(data['distance_matrix'], list):
            data['distance_matrix'] = np.array(data['distance_matrix'])
        
        if 'vectors' in data and isinstance(data['vectors'], list):
            data['vectors'] = [np.array(v) for v in data['vectors']]
        
        return data
    
    def generate_summary_report(self, pipeline_results: Dict) -> str:
        """Generate a human-readable summary report"""
        
        report = []
        report.append("🎯 REWARD EVOLUTION PIPELINE SUMMARY")
        report.append("=" * 50)
        report.append("")
        
        # Basic info
        summary = pipeline_results.get('summary', {})
        report.append(f"📊 Total Duration: {pipeline_results.get('pipeline_duration', 'Unknown')}")
        report.append(f"🔄 Iterations Completed: {summary.get('iterations_completed', 0)}")
        report.append(f"🎯 Total Rewards Generated: {summary.get('total_rewards_generated', 0)}")
        report.append("")
        
        # Evolution metrics
        if 'evolution_metrics' in summary:
            metrics = summary['evolution_metrics']
            report.append("📈 EVOLUTION METRICS")
            report.append("-" * 30)
            
            if 'diversity_trend' in metrics and metrics['diversity_trend']:
                trend = metrics['diversity_trend']
                report.append(f"🌟 Diversity Evolution: {trend[0]:.3f} → {trend[-1]:.3f}")
                
            if 'cluster_count_trend' in metrics and metrics['cluster_count_trend']:
                trend = metrics['cluster_count_trend']
                report.append(f"🎯 Cluster Count Evolution: {trend[0]} → {trend[-1]}")
            
            report.append("")
        
        # Iteration details
        iterations = pipeline_results.get('iterations', [])
        if iterations:
            report.append("🔄 ITERATION DETAILS")
            report.append("-" * 30)
            
            for i, iteration in enumerate(iterations, 1):
                report.append(f"Iteration {i}:")
                report.append(f"  ⏱️  Duration: {iteration.get('duration', 'Unknown')}")
                report.append(f"  🎯 Rewards Generated: {len(iteration.get('reward_files', []))}")
                
                clusters = iteration.get('clusters', {})
                report.append(f"  🎪 Clusters Found: {len(clusters)}")
                
                selected = iteration.get('selected_rewards', {})
                if selected:
                    closest = selected.get('closest', [])
                    furthest = selected.get('furthest', [])
                    report.append(f"  🎯 Selected Closest: {len(closest)}")
                    report.append(f"  🚀 Selected Furthest: {len(furthest)}")
                
                report.append("")
        
        # Final clusters
        final_clusters = summary.get('final_clusters', {})
        if final_clusters:
            report.append("🎪 FINAL CLUSTERING RESULTS")
            report.append("-" * 30)
            
            for cluster_id, members in final_clusters.items():
                report.append(f"Cluster {cluster_id}: {len(members)} members")
                for member in members:
                    report.append(f"  - {member}")
            
            report.append("")
        
        return "\n".join(report)
    
    def validate_pipeline_state(self, iteration: int, expected_files: int = 16) -> Dict[str, bool]:
        """Validate the state of the pipeline at a given iteration"""
        
        validation = {
            'api_key_present': bool(os.getenv('OPENAI_API_KEY')),
            'directories_exist': True,
            'previous_iteration_complete': True,
            'expected_file_count': True
        }
        
        # Check directories
        required_dirs = ['starc/llm', 'starc/rewards', 'starc/results']
        for dir_path in required_dirs:
            if not pathlib.Path(dir_path).exists():
                validation['directories_exist'] = False
                break
        
        # Check previous iteration (if not first)
        if iteration > 1:
            prev_results_dir = pathlib.Path(f"starc/results/pipeline/iteration_{iteration-1}")
            if not prev_results_dir.exists():
                validation['previous_iteration_complete'] = False
        
        return validation
    
    def cleanup_temp_files(self, temp_dir: pathlib.Path):
        """Clean up temporary files created during pipeline execution"""
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"  🧹 Cleaned up temporary files in {temp_dir}")
    
    def estimate_pipeline_duration(self, n_iterations: int = 4, n_rewards_per_iteration: int = 16) -> str:
        """Estimate total pipeline duration based on historical performance"""
        
        # Based on previous runs: ~18 minutes per reward for STARC analysis
        minutes_per_reward = 18
        llm_minutes_per_iteration = 2  # LLM calls are relatively fast
        
        total_rewards = n_iterations * n_rewards_per_iteration
        estimated_minutes = (total_rewards * minutes_per_reward) + (n_iterations * llm_minutes_per_iteration)
        
        hours = estimated_minutes // 60
        minutes = estimated_minutes % 60
        
        if hours > 0:
            return f"~{hours}h {minutes}m"
        else:
            return f"~{minutes}m"
    
    def create_iteration_summary(self, iteration_data: Dict) -> Dict:
        """Create a concise summary for a single iteration"""
        
        summary = {
            'iteration': iteration_data.get('iteration', 0),
            'duration': iteration_data.get('duration', 'Unknown'),
            'rewards_generated': len(iteration_data.get('reward_files', [])),
            'clusters_found': len(iteration_data.get('clusters', {})),
            'timestamp': iteration_data.get('timestamp', 'Unknown')
        }
        
        # Add selection info if available
        selected = iteration_data.get('selected_rewards', {})
        if selected:
            summary['closest_selected'] = selected.get('closest', [])
            summary['furthest_selected'] = selected.get('furthest', [])
            
            # Add distance info if available
            if 'closest_distances' in selected:
                summary['avg_closest_distance'] = np.mean(selected['closest_distances'])
            if 'furthest_distances' in selected:
                summary['avg_furthest_distance'] = np.mean(selected['furthest_distances'])
        
        return summary
    
    def format_distance_matrix_for_prompt(self, distance_matrix: np.ndarray, reward_names: List[str], max_size: int = 32) -> str:
        """
        Format a (possibly large) STARC distance matrix and reward names as a readable string for LLM prompt inclusion.
        If the matrix is larger than max_size, only show the top-left max_size x max_size block and mention truncation.
        """
        n = len(reward_names)
        show_n = min(n, max_size)
        lines = []
        lines.append("STARC Distance Matrix (rows=from, cols=to; truncated to {}x{} if needed)".format(show_n, show_n))
        lines.append("=" * 60)
        lines.append(f"Matrix size: {n} x {n}")
        lines.append(f"Reward functions: {n}")
        lines.append("")
        # Write column headers
        header = "     " + "".join(f"{i:>7d}" for i in range(show_n))
        lines.append(header)
        for i in range(show_n):
            row = f"{i:2d}: " + "".join(f"{distance_matrix[i, j]:7.3f}" for j in range(show_n)) + f"  # {reward_names[i]}"
            lines.append(row)
        if n > max_size:
            lines.append(f"... (truncated, showing only first {max_size} rewards)")
        # Summary statistics
        lines.append("")
        lines.append("Summary Statistics:")
        lines.append(f"Mean distance: {np.mean(distance_matrix):.4f}")
        lines.append(f"Std distance:  {np.std(distance_matrix):.4f}")
        lines.append(f"Min distance:  {np.min(distance_matrix):.4f}")
        lines.append(f"Max distance:  {np.max(distance_matrix):.4f}")
        # Pairwise (excluding diagonal)
        if n > 1:
            upper_triangle = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
            lines.append(f"Mean pairwise distance: {np.mean(upper_triangle):.4f}")
        return "\n".join(lines) 