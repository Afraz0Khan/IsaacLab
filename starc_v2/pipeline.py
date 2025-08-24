#!/usr/bin/env python3
"""
STARC v2 - LLM-Driven Reward Evolution Pipeline

Key differences from v1:
- LLM selects rewards based on full STARC matrix
- No automated closest/furthest selection
- Chat history maintained across iterations
- Full distance matrix including reference bounds provided to LLM
"""

import os
import json
import numpy as np
import pathlib
from typing import List, Dict, Tuple
from datetime import datetime

# Import components
from .core.starc_analyzer import STARCv2Analyzer
from .llm.reward_generator import STARCv2RewardGenerator
from .config import STARCv2Config

class STARCv2Pipeline:
    """
    STARC v2 Pipeline - LLM-Driven Reward Evolution
    
    This pipeline lets the LLM make all selection decisions based on 
    comprehensive STARC distance information.
    """
    
    def __init__(self, base_dir: str = None):
        # Auto-detect base directory
        if base_dir is None:
            current_dir = pathlib.Path.cwd()
            if current_dir.name == "starc_v2":
                self.base_dir = current_dir
            else:
                self.base_dir = current_dir / "starc_v2"
        else:
            self.base_dir = pathlib.Path(base_dir)
            
        self.results_dir = self.base_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.analyzer = STARCv2Analyzer()
        self.generator = STARCv2RewardGenerator()
        
        # Pipeline state
        self.global_reward_counter = 0
        self.all_iteration_results = []
        self.cumulative_rewards: List[pathlib.Path] = []
        # Cache of previously computed reward vectors (exclude references)
        self._cached_reward_names: List[str] = []
        self._cached_reward_vectors: List[np.ndarray] = []
        # Reference vectors (GROUND_TRUTH and NEGATIVE_GROUND_TRUTH), set on first computation
        self._reference_names: List[str] = []
        self._reference_vectors: List[np.ndarray] = []
        
        # Precompute a single transition batch for consistent STARC distances
        from .config import STARCv2Config
        print(f"[INIT] Creating dummy environment for transition batch (env={STARCv2Config.ENV_NAME})...")
        if STARCv2Config.ENV_NAME == 'halfcheetah':
            from starc.rewards.ground_truth_reward import GroundTruthReward
            from starc_v2.core.half_cheetah_env_fixed import HalfCheetahEnvFixed as HalfCheetahEnv
            # IMPORTANT: no SARSA needed to build a transition batch → set n_episodes_sarsa=0
            dummy_env = HalfCheetahEnv(GroundTruthReward(), STARCv2Config.DISCOUNT, 0)
            print("[INIT] Building transition batch (this can take ~seconds on first run)...")
            self._transition_batch = self.analyzer._build_transition_batch(dummy_env)  # pylint: disable=protected-access
            print("[INIT] Transition batch ready.")
        else:
            # Gymnasium-based transition batch for Ant/Humanoid
            print("[INIT] Building transition batch via Gymnasium (Ant/Humanoid)...")
            from .core.env_transitions import build_transition_batch
            self._transition_batch = build_transition_batch(STARCv2Config.ENV_NAME, n_samples=STARCv2Config.N_SAMPLES)
            print("[INIT] Transition batch ready.")
        
    def run_complete_pipeline(self) -> Dict:
        """Run the complete STARC v2 pipeline"""
        print("🚀 Starting STARC v2 - LLM-Driven Reward Evolution Pipeline")
        print("=" * 70)
        
        # Print configuration
        STARCv2Config.print_config()
        
        pipeline_start = datetime.now()
        
        # Run iterations
        for iteration in range(1, STARCv2Config.TOTAL_ITERATIONS + 1):
            print(f"\n🔄 ITERATION {iteration}/{STARCv2Config.TOTAL_ITERATIONS}")
            print("-" * 50)
            
            iteration_result = self.run_single_iteration(iteration)
            self.all_iteration_results.append(iteration_result)
            
            # Save iteration results
            self.save_iteration_results(iteration, iteration_result)
            
            # Save chat history after each iteration
            self.generator.save_chat_history(iteration)
            
        pipeline_end = datetime.now()
        
        # Compile final results
        final_results = self.compile_final_results(pipeline_start, pipeline_end)
        
        # Save final results
        final_path = self.results_dir / STARCv2Config.ENV_NAME / "final_pipeline_results.json"
        with open(final_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=self._json_serializer)
            
        print(f"\n🎉 STARC v2 Pipeline Complete!")
        print(f"📊 Final results saved to: {final_path}")
        print(f"⏱️  Total duration: {final_results['pipeline_duration']}")
        print(f"🧠 Chat history maintained with {self.generator.get_chat_history_summary()['total_messages']} messages")
        
        # ------------------------------------------------------------------
        # Cluster all rewards (excluding references) using final distance matrix
        # ------------------------------------------------------------------
        final_starcs = self.all_iteration_results[-1]['starc_results']
        dist = final_starcs['distance_matrix']
        names = final_starcs['reward_names']

        # simple union-find clustering
        clusters = self._cluster_rewards(dist, names, STARCv2Config.CLUSTER_THRESHOLD)

        cluster_path = self.results_dir / STARCv2Config.ENV_NAME / "final_clusters.json"
        with open(cluster_path, 'w') as f:
            json.dump(clusters, f, indent=2, default=self._json_serializer)
        print(f"🔗 Saved final reward clusters → {cluster_path}  (k={len(clusters)})")
        
        return final_results
    
    def run_single_iteration(self, iteration: int) -> Dict:
        """Run a single iteration of the pipeline"""
        iteration_start = datetime.now()
        
        if iteration == 1:
            # First iteration: generate initial rewards
            print(f"📝 Step 1: Generating initial {STARCv2Config.REWARDS_PER_ITERATION} reward functions...")
            
            start_number = self.global_reward_counter + 1
            reward_files = self.generator.generate_initial_rewards(iteration, start_number)
            self.global_reward_counter += len(reward_files)
            
            # Update cumulative list ---------------------------------------
            self.cumulative_rewards.extend(reward_files)
            
            selected_rewards = []  # No selection in first iteration
            
        else:
            # Subsequent iterations: LLM selects and generates
            print(f"📊 Step 1: Analyzing previous iteration results...")
            
            # Get previous iteration results
            prev_results = self.all_iteration_results[-1]
            
            # Format STARC matrix for LLM (using compact format to save tokens)
            starc_matrix_str = self.analyzer.format_matrix_for_llm(
                prev_results['starc_results'], 
                compact=STARCv2Config.USE_COMPACT_MATRIX
            )
            
            print(f"🤖 Step 2: LLM selecting rewards and generating new ones...")
            
            start_number = self.global_reward_counter + 1
            selected_rewards, reward_files = self.generator.generate_rewards_with_selection(
                iteration, starc_matrix_str, start_number
            )
            self.global_reward_counter += len(reward_files)
            
            self.cumulative_rewards.extend(reward_files)
        
        # Analyze only the new rewards for this iteration to avoid retraining old ones
        print(f"🔬 Step 3: Running STARC analysis for {len(reward_files)} new rewards (incremental mode)...")
        # Include reference rewards only on the first iteration to cache them once
        new_results = self.analyzer.analyze_rewards(
            reward_files,
            iteration,
            precomputed_batch=self._transition_batch,
            include_references=(iteration == 1),
        )

        # Extract reference indices and split out references vs new rewards
        ref_idx = new_results.get("reference_indices", {"ground_truth": 0, "negative_ground_truth": 1})
        names_all = new_results["reward_names"]
        vecs_all = new_results["vectors"]

        # Ensure reference vectors are cached once
        if not self._reference_names:
            # Expect references to be named exactly as below per analyzer
            gt_name = "GROUND_TRUTH"
            neg_name = "NEGATIVE_GROUND_TRUTH"
            # Try to fetch by provided indices; if not found, fall back to lookup by name
            try:
                gt_vec = vecs_all[ref_idx["ground_truth"]]
                neg_vec = vecs_all[ref_idx["negative_ground_truth"]]
            except Exception:
                # Fallback by name search
                gt_idx = names_all.index(gt_name) if gt_name in names_all else 0
                neg_idx = names_all.index(neg_name) if neg_name in names_all else 1
                gt_vec = vecs_all[gt_idx]
                neg_vec = vecs_all[neg_idx]
            self._reference_names = [gt_name, neg_name]
            self._reference_vectors = [np.array(gt_vec), np.array(neg_vec)]

        # Collect new reward vectors (exclude references)
        non_ref_indices = [i for i, n in enumerate(names_all) if n not in ("GROUND_TRUTH", "NEGATIVE_GROUND_TRUTH")]
        new_names_only = [names_all[i] for i in non_ref_indices]
        new_vecs_only = [np.array(vecs_all[i]) for i in non_ref_indices]

        # Update caches
        self._cached_reward_names.extend(new_names_only)
        self._cached_reward_vectors.extend(new_vecs_only)

        # Build combined names and vectors: references first, then all cached rewards
        combined_names = self._reference_names + self._cached_reward_names
        combined_vectors = self._reference_vectors + self._cached_reward_vectors

        # Compute full distance matrix incrementally (Euclidean distances)
        N = len(combined_vectors)
        distance_matrix = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            vi = np.asarray(combined_vectors[i])
            for j in range(i + 1, N):
                vj = np.asarray(combined_vectors[j])
                d = float(np.linalg.norm(vi - vj))
                distance_matrix[i, j] = d
                distance_matrix[j, i] = d

        # Indices for references
        reference_indices = {"ground_truth": 0, "negative_ground_truth": 1}
        gt_distances = distance_matrix[reference_indices["ground_truth"], :].tolist()
        neg_gt_distances = distance_matrix[reference_indices["negative_ground_truth"], :].tolist()

        # Assemble starc_results in the same schema expected elsewhere
        starc_results = {
            "reward_names": combined_names,
            "vectors": combined_vectors,
            "distance_matrix": distance_matrix,
            "ground_truth_distances": gt_distances,
            "negative_ground_distances": neg_gt_distances,
            "reference_indices": reference_indices,
            "transition_data": new_results.get("transition_data"),
            "iteration": iteration,
        }
        
        # Format matrix for this iteration
        matrix_str = self.analyzer.format_matrix_for_llm(
            starc_results, 
            compact=STARCv2Config.USE_COMPACT_MATRIX
        )
        
        iteration_end = datetime.now()
        
        return {
            "iteration": iteration,
            "duration": str(iteration_end - iteration_start),
            "start_time": iteration_start.isoformat(),
            "end_time": iteration_end.isoformat(),
            "reward_files": [str(f) for f in reward_files],
            "selected_rewards": selected_rewards,
            "starc_results": starc_results,
            "matrix_string": matrix_str,
            "global_reward_counter": self.global_reward_counter,
            "chat_history_summary": self.generator.get_chat_history_summary()
        }
    
    def save_iteration_results(self, iteration: int, results: Dict):
        """Save results for a single iteration"""
        # Env-specific result subdir to avoid confusion across multiple runs
        iteration_dir = self.results_dir / STARCv2Config.ENV_NAME / f"iteration_{iteration}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_path = iteration_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=self._json_serializer)
        
        # Save distance matrix as numpy array
        distance_matrix = results['starc_results']['distance_matrix']
        matrix_path = iteration_dir / "distance_matrix.npy"
        np.save(matrix_path, distance_matrix)
        
        # Save distance matrix as text
        matrix_txt_path = iteration_dir / "distance_matrix.txt"
        with open(matrix_txt_path, 'w') as f:
            f.write(results['matrix_string'])
        
        # Save reward names
        names_path = iteration_dir / "reward_names.json"
        with open(names_path, 'w') as f:
            json.dump(results['starc_results']['reward_names'], f, indent=2)
        
        print(f"  💾 Saved iteration {iteration} results to {iteration_dir}")
    
    def compile_final_results(self, start_time: datetime, end_time: datetime) -> Dict:
        """Compile final pipeline results"""
        chat_summary = self.generator.get_chat_history_summary()
        
        # Calculate some statistics
        total_rewards_generated = self.global_reward_counter
        total_rewards_analyzed = sum(
            len(result['starc_results']['reward_names']) - 2  # Exclude reference rewards
            for result in self.all_iteration_results
        )
        
        # Get final distance matrix statistics
        if self.all_iteration_results:
            final_results = self.all_iteration_results[-1]['starc_results']
            final_gt_distances = final_results['ground_truth_distances'][2:]  # Exclude references
            final_neg_gt_distances = final_results['negative_ground_distances'][2:]  # Exclude references
            
            distance_stats = {
                "final_gt_distance_stats": {
                    "min": float(np.min(final_gt_distances)),
                    "max": float(np.max(final_gt_distances)),
                    "mean": float(np.mean(final_gt_distances)),
                    "std": float(np.std(final_gt_distances))
                },
                "final_neg_gt_distance_stats": {
                    "min": float(np.min(final_neg_gt_distances)),
                    "max": float(np.max(final_neg_gt_distances)),
                    "mean": float(np.mean(final_neg_gt_distances)),
                    "std": float(np.std(final_neg_gt_distances))
                }
            }
        else:
            distance_stats = {}
        
        return {
            "pipeline_version": "2.0.0",
            "pipeline_type": "LLM-Driven Reward Evolution",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "pipeline_duration": str(end_time - start_time),
            "configuration": {
                "total_iterations": STARCv2Config.TOTAL_ITERATIONS,
                "rewards_per_iteration": STARCv2Config.REWARDS_PER_ITERATION,
                "rewards_to_select": STARCv2Config.REWARDS_TO_SELECT,
                "n_episodes_sarsa": STARCv2Config.N_EPISODES_SARSA,
                "primary_model": STARCv2Config.PRIMARY_MODEL,
                "temperature": STARCv2Config.TEMPERATURE
            },
            "statistics": {
                "total_rewards_generated": total_rewards_generated,
                "total_rewards_analyzed": total_rewards_analyzed,
                "total_iterations_completed": len(self.all_iteration_results),
                **distance_stats
            },
            "chat_history_summary": chat_summary,
            "iterations": self.all_iteration_results,
            "llm_selections": self.extract_llm_selections(),
            "evolution_analysis": self.analyze_evolution_patterns()
        }
    
    def extract_llm_selections(self) -> List[Dict]:
        """Extract LLM selection decisions across iterations"""
        selections = []
        
        for result in self.all_iteration_results:
            if result['selected_rewards']:  # Skip first iteration
                selections.append({
                    "iteration": result['iteration'],
                    "selected_rewards": result['selected_rewards'],
                    "selection_count": len(result['selected_rewards']),
                    "total_available": len(result['starc_results']['reward_names']) - 2  # Exclude references
                })
        
        return selections
    
    def analyze_evolution_patterns(self) -> Dict:
        """Analyze evolution patterns across iterations"""
        if len(self.all_iteration_results) < 2:
            return {"note": "Need at least 2 iterations for evolution analysis"}
        
        # Track distance evolution
        gt_distance_evolution = []
        neg_gt_distance_evolution = []
        
        for result in self.all_iteration_results:
            gt_dists = result['starc_results']['ground_truth_distances'][2:]  # Exclude references
            neg_gt_dists = result['starc_results']['negative_ground_distances'][2:]  # Exclude references
            
            if len(gt_dists) == 0 or len(neg_gt_dists) == 0:
                # Gracefully handle empty arrays (e.g., iteration produced no valid rewards)
                gt_distance_evolution.append({
                    "iteration": result['iteration'],
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None
                })
                neg_gt_distance_evolution.append({
                    "iteration": result['iteration'],
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None
                })
                continue

            gt_distance_evolution.append({
                "iteration": result['iteration'],
                "min": float(np.min(gt_dists)),
                "max": float(np.max(gt_dists)),
                "mean": float(np.mean(gt_dists)),
                "std": float(np.std(gt_dists))
            })
            
            neg_gt_distance_evolution.append({
                "iteration": result['iteration'],
                "min": float(np.min(neg_gt_dists)),
                "max": float(np.max(neg_gt_dists)),
                "mean": float(np.mean(neg_gt_dists)),
                "std": float(np.std(neg_gt_dists))
            })
        
        return {
            "ground_truth_distance_evolution": gt_distance_evolution,
            "negative_ground_truth_distance_evolution": neg_gt_distance_evolution,
            "diversity_trends": self.calculate_diversity_trends(),
            "selection_patterns": self.analyze_selection_patterns()
        }
    
    def calculate_diversity_trends(self) -> List[Dict]:
        """Calculate diversity trends across iterations"""
        diversity_trends = []
        
        for result in self.all_iteration_results:
            matrix = result['starc_results']['distance_matrix']
            # Exclude reference rewards from diversity calculation
            non_ref_matrix = matrix[2:, 2:]
            
            if non_ref_matrix.size > 0:
                # Calculate various diversity metrics
                upper_triangle = non_ref_matrix[np.triu_indices_from(non_ref_matrix, k=1)]
                
                diversity_trends.append({
                    "iteration": result['iteration'],
                    "mean_pairwise_distance": float(np.mean(upper_triangle)),
                    "min_pairwise_distance": float(np.min(upper_triangle)),
                    "max_pairwise_distance": float(np.max(upper_triangle)),
                    "diversity_std": float(np.std(upper_triangle))
                })
        
        return diversity_trends
    
    def analyze_selection_patterns(self) -> Dict:
        """Analyze LLM selection patterns"""
        if len(self.all_iteration_results) < 2:
            return {"note": "Need at least 2 iterations for selection analysis"}
        
        # Analyze what types of rewards the LLM tends to select
        selection_analysis = {
            "selection_consistency": [],
            "distance_preferences": [],
            "diversity_preferences": []
        }
        
        for i, result in enumerate(self.all_iteration_results[1:], 1):  # Skip first iteration
            selected = result['selected_rewards']
            starc_results = result['starc_results']
            
            # Find indices of selected rewards
            selected_indices = []
            for reward_name in selected:
                try:
                    idx = starc_results['reward_names'].index(reward_name)
                    selected_indices.append(idx)
                except ValueError:
                    continue
            
            if selected_indices:
                # Analyze distance preferences
                selected_gt_distances = [starc_results['ground_truth_distances'][idx] for idx in selected_indices]
                all_gt_distances = starc_results['ground_truth_distances'][2:]  # Exclude references
                
                selection_analysis["distance_preferences"].append({
                    "iteration": i + 1,
                    "selected_mean_gt_distance": float(np.mean(selected_gt_distances)),
                    "all_mean_gt_distance": float(np.mean(all_gt_distances)),
                    "preference_bias": float(np.mean(selected_gt_distances) - np.mean(all_gt_distances))
                })
        
        return selection_analysis
    
    def _json_serializer(self, obj):
        """JSON serializer for numpy arrays and other objects"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, pathlib.Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # ------------------------------------------------------------------
    # Helper: cluster by distance threshold
    # ------------------------------------------------------------------
    @staticmethod
    def _cluster_rewards(distance_matrix, names, threshold: float):
        """Hybrid clustering: agglomerative with distance threshold, then KMeans refinement.

        - Excludes reference rewards (GROUND_TRUTH, NEGATIVE_GROUND_TRUTH)
        - First, cut an agglomerative tree at the given distance threshold
        - Then, run a small KMeans on each cluster with > 2× median size to split overly large clusters
        """
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering, KMeans
        from .config import STARCv2Config

        # Exclude references
        indices = [i for i, n in enumerate(names) if n not in ("GROUND_TRUTH", "NEGATIVE_GROUND_TRUTH")]
        if not indices:
            return {}

        sub_mat = distance_matrix[np.ix_(indices, indices)]
        sub_names = [names[i] for i in indices]

        # Agglomerative: choose mode based on config (sklearn version compatible)
        if STARCv2Config.CLUSTER_N_CLUSTERS is not None:
            try:
                agg = AgglomerativeClustering(
                    n_clusters=int(STARCv2Config.CLUSTER_N_CLUSTERS),
                    linkage="average",
                    metric="precomputed",
                )
            except TypeError:
                # Older sklearn: use 'affinity'
                agg = AgglomerativeClustering(
                    n_clusters=int(STARCv2Config.CLUSTER_N_CLUSTERS),
                    linkage="average",
                    affinity="precomputed",
                )
        else:
            try:
                agg = AgglomerativeClustering(
                    n_clusters=None,
                    linkage="average",
                    metric="precomputed",
                    distance_threshold=threshold,
                    compute_full_tree=True,
                )
            except TypeError:
                agg = AgglomerativeClustering(
                    n_clusters=None,
                    linkage="average",
                    affinity="precomputed",
                    distance_threshold=threshold,
                    compute_full_tree=True,
                )
        # Guard against NaNs/Infs in distance matrix before clustering
        sub_mat = np.nan_to_num(sub_mat, nan=0.0, posinf=0.0, neginf=0.0)
        labels = agg.fit_predict(sub_mat)

        # Group by label
        clusters = {}
        for name, lab in zip(sub_names, labels):
            clusters.setdefault(int(lab), []).append(name)

        # Optional refinement: split very large clusters by KMeans in vector space is unavailable here,
        # so we split using distance-space MDS approximation (skip for simplicity). Keep clusters as-is.

        return {f"cluster_{k}": v for k, v in enumerate(clusters.values())}


def main():
    """Main entry point for the pipeline"""
    pipeline = STARCv2Pipeline()
    results = pipeline.run_complete_pipeline()
    
    print(f"\n🎯 Pipeline Summary:")
    print(f"   Total rewards generated: {results['statistics']['total_rewards_generated']}")
    print(f"   Total chat messages: {results['chat_history_summary']['total_messages']}")
    print(f"   Iterations completed: {results['statistics']['total_iterations_completed']}")
    
    return results


if __name__ == "__main__":
    main() 