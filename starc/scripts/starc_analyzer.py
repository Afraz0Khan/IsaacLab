import importlib.util
import pathlib
import numpy as np
from typing import List, Dict, Tuple
from sklearn.cluster import DBSCAN
from functools import partial

from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.rewards.ground_truth_reward import GroundTruthReward

class STARCAnalyzer:
    def __init__(self):
        self.discount = 0.848
        self.n_episode = 6000
        self.n_samples = 96
        self.cluster_eps = 0.15
        
    def analyze_rewards(self, reward_files: List[pathlib.Path], iteration: int) -> Dict:
        """Analyze reward functions using STARC distance"""
        print(f"  🔬 Analyzing {len(reward_files)} reward functions...")
        
        # Build environment and sample transitions
        dummy_env = HalfCheetahEnv(GroundTruthReward(), self.discount, self.n_episode)
        S, A, SP = self._build_transition_batch(dummy_env)
        
        # Process each reward function
        vectors, names, ground_truth_vector = self._process_rewards(reward_files, S, A, SP)
        
        # Calculate distance matrix
        distance_matrix = self._calculate_distance_matrix(vectors)
        
        # Calculate distances to ground truth
        gt_distances = self._calculate_ground_truth_distances(vectors, ground_truth_vector)
        
        return {
            "reward_names": names,
            "vectors": vectors,
            "distance_matrix": distance_matrix,
            "ground_truth_distances": gt_distances,
            "ground_truth_vector": ground_truth_vector,
            "transition_data": {"S": S, "A": A, "SP": SP}
        }
    
    def cluster_rewards(self, distance_matrix: np.ndarray, reward_names: List[str], eps: float = None) -> Dict:
        """Cluster rewards based on distance matrix"""
        if eps is None:
            eps = self.cluster_eps
            
        print(f"  🎯 Clustering with eps={eps}...")
        
        # Check if we have any rewards to cluster
        if distance_matrix.shape[0] == 0:
            print("  ⚠️  No rewards to cluster - all reward functions failed to process")
            return {}
        
        # Run DBSCAN clustering
        model = DBSCAN(eps=eps, metric="precomputed", min_samples=1).fit(distance_matrix)
        
        # Organize results
        clusters = {}
        for idx, cluster_id in enumerate(model.labels_):
            clusters.setdefault(int(cluster_id), []).append(reward_names[idx])
        
        return clusters
    
    def _build_transition_batch(self, env, n=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a batch of transitions for STARC analysis"""
        if n is None:
            n = self.n_samples
            
        s_list, a_list, sp_list = [], [], []
        obs, _ = env.reset()
        
        for i in range(n):
            a = env.action_space.sample()
            s = obs.copy()
            obs, *_ = env.step(a)[:4]
            sp = obs.copy()
            
            s_list.append(s)
            a_list.append(a)
            sp_list.append(sp)
            
            # Reset periodically to avoid getting stuck
            if i % 100 == 0:
                obs, _ = env.reset()
        
        return np.vstack(s_list), np.vstack(a_list), np.vstack(sp_list)
    
    def _process_rewards(self, reward_files: List[pathlib.Path], S: np.ndarray, A: np.ndarray, SP: np.ndarray) -> Tuple[List[np.ndarray], List[str], np.ndarray]:
        """Process reward functions and compute their vectors"""
        vectors = []
        names = []
        
        # First, compute ground truth vector
        gt_env = HalfCheetahEnv(GroundTruthReward(), self.discount, self.n_episode)
        gt_vector = self._compute_reward_vector(gt_env, S, A, SP)
        
        # Process each reward file
        for reward_file in reward_files:
            try:
                print(f"    📊 Processing {reward_file.name}...")
                
                # Load reward function
                reward_obj = self._load_reward_from_file(reward_file)
                print(f"    ✅ Loaded reward class: {type(reward_obj).__name__}")
                
                # Create environment
                env = HalfCheetahEnv(reward_obj, self.discount, self.n_episode)
                
                # Compute vector
                vector = self._compute_reward_vector(env, S, A, SP)
                print(f"    ✅ Computed vector shape: {vector.shape}")
                
                vectors.append(vector)
                names.append(reward_file.stem)
                
            except Exception as e:
                import traceback
                print(f"    ❌ Failed to process {reward_file.name}: {e}")
                print(f"    🔍 Traceback: {traceback.format_exc()}")
                continue
        
        print(f"  ✅ Successfully processed {len(vectors)}/{len(reward_files)} rewards")
        return vectors, names, gt_vector
    
    def _load_reward_from_file(self, path: pathlib.Path):
        """Load reward function from Python file"""
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None:
            raise ImportError(f"Failed to create module spec for {path}")
        
        if spec.loader is None:
            raise ImportError(f"No loader found for module spec {path}")
            
        module = importlib.util.module_from_spec(spec)
        if module is None:
            raise ImportError(f"Failed to create module from spec for {path}")
        
        spec.loader.exec_module(module)
        
        # Find the reward class dynamically
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and 
                hasattr(obj, '__bases__') and 
                any('RewardFunc' in str(base) for base in obj.__bases__)):
                return obj()
        
        # Fallback - try different naming patterns
        possible_names = ["RewardFunc", f"RewardFunc_{path.stem.split('_')[-1]}", f"RewardFunc{path.stem.split('_')[-1]}"]
        for name in possible_names:
            if hasattr(module, name):
                return getattr(module, name)()
        
        raise ImportError(f"Could not find a valid RewardFunc class in {path}")
    
    def _compute_reward_vector(self, env, S: np.ndarray, A: np.ndarray, SP: np.ndarray) -> np.ndarray:
        """Compute the reward vector for a given environment"""
        try:
            # Create canonical function (simplified version)
            canon_f = self._canon_callable(env.reward_func_curried)
            
            # Compute vector
            vec = canon_f(S, A, SP)
            if vec is None:
                raise ValueError("Canon function returned None")
            
            vec = vec.astype(np.float32)
            
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec /= norm
            
            return vec
        except Exception as e:
            raise ValueError(f"Failed to compute reward vector: {e}")
    
    def _canon_callable(self, reward_callable):
        """Create a canonical callable for the reward function"""
        def vectorized_reward(S, A, SP):
            """Vectorized version that processes arrays of states/actions"""
            results = []
            for i, (s, a, sp) in enumerate(zip(S, A, SP)):
                try:
                    r = reward_callable(s, a, sp)
                    if r is None:
                        raise ValueError(f"Reward function returned None for transition {i}")
                    results.append(r)
                except Exception as e:
                    raise ValueError(f"Error computing reward for transition {i}: {e}")
            return np.array(results)
        
        return vectorized_reward
    
    def _calculate_distance_matrix(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Calculate pairwise distance matrix between reward vectors"""
        n = len(vectors)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance_matrix[i, j] = np.linalg.norm(vectors[i] - vectors[j])
        
        return distance_matrix
    
    def _calculate_ground_truth_distances(self, vectors: List[np.ndarray], gt_vector: np.ndarray) -> List[float]:
        """Calculate distances from each reward to ground truth"""
        distances = []
        for vector in vectors:
            distance = np.linalg.norm(vector - gt_vector)
            distances.append(float(distance))
        
        return distances 