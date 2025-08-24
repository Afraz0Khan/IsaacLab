import importlib.util
import pathlib
import numpy as np
from typing import List, Dict, Tuple, Optional
from functools import partial
import os
import torch

# Import from the original starc directory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'starc'))

from starc.rewards.ground_truth_reward import GroundTruthReward
from ..rewards.negative_ground_reward_fixed import NegativeGroundRewardFixed as NegativeGroundReward
from .half_cheetah_env_fixed import HalfCheetahEnvFixed as HalfCheetahEnv
from ..rewards.ground_truth_ant import GroundTruthAntReward, NegativeGroundAntReward
from ..rewards.ground_truth_humanoid import GroundTruthHumanoidReward, NegativeGroundHumanoidReward

from ..config import STARCv2Config

class STARCv2Analyzer:
    """
    STARC analyzer for v2 pipeline.
    
    Key differences from v1:
    - Includes ground truth and negative ground truth in distance matrix
    - Provides comprehensive distance information to LLM
    - No automatic clustering or selection
    """
    
    def __init__(self):
        self.discount = STARCv2Config.DISCOUNT
        self.n_episodes_sarsa = STARCv2Config.N_EPISODES_SARSA
        self.n_samples = STARCv2Config.N_SAMPLES
        
    def analyze_rewards(self, reward_files: List[pathlib.Path], iteration: int, precomputed_batch: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None, include_references: bool = True) -> Dict:
        """
        Analyze reward functions using STARC distance. Optionally include reference rewards.
        
        Returns:
            Dict containing:
            - reward_names: List of all reward names (including references)
            - distance_matrix: Full distance matrix including references
            - ground_truth_distances: Distances to ground truth
            - negative_ground_distances: Distances to negative ground truth
            - reference_indices: Indices of reference rewards in the matrix
        """
        print(f"  🔬 Analyzing {len(reward_files)} reward functions{' with reference bounds' if include_references else ''}...")
        
        # Build or reuse transition batch
        if precomputed_batch is None:
            print("  [STARc] No precomputed batch passed; creating dummy env and sampling transitions…")
            print("  [STARC] Building dummy environment (no SARSA)…")
            dummy_env = HalfCheetahEnv(GroundTruthReward(), self.discount, 0)
            S, A, SP, X_VEL = self._build_transition_batch(dummy_env)
        else:
            print("  [STARC] Using precomputed transition batch.")
            S, A, SP, X_VEL = precomputed_batch
        
        # Process reward functions
        if include_references:
            vectors, names, reference_indices = self._process_rewards_with_references(reward_files, S, A, SP, X_VEL)
        else:
            vectors, names = self._process_rewards_only(reward_files, S, A, SP, X_VEL)
            reference_indices = None
        
        # Calculate distance matrix among processed rewards
        distance_matrix = self._calculate_distance_matrix(vectors)
        
        result: Dict = {
            "reward_names": names,
            "vectors": vectors,
            "distance_matrix": distance_matrix,
            "transition_data": {"S": S, "A": A, "SP": SP, "X_VEL": X_VEL},
            "iteration": iteration,
        }
        if include_references and reference_indices is not None:
            gt_index = reference_indices['ground_truth']
            neg_gt_index = reference_indices['negative_ground_truth']
            result.update({
                "ground_truth_distances": distance_matrix[gt_index, :].tolist(),
                "negative_ground_distances": distance_matrix[neg_gt_index, :].tolist(),
                "reference_indices": reference_indices,
            })
        return result
    
    def format_matrix_for_llm(self, results: Dict, compact: bool = True) -> str:
        """
        Format the STARC distance matrix for LLM consumption.
        
        Args:
            results: STARC analysis results
            compact: If True, use upper triangle only (saves ~50% tokens)
        
        Creates a comprehensive, readable representation of all distances
        including relationships to reference rewards.
        """
        if compact:
            return self._format_compact_matrix(results)
        else:
            return self._format_full_matrix(results)
    
    def _format_compact_matrix(self, results: Dict) -> str:
        """Format using upper triangle only - much more token efficient"""
        names = results['reward_names']
        matrix = results['distance_matrix']
        gt_distances = results['ground_truth_distances']
        neg_gt_distances = results['negative_ground_distances']
        ref_indices = results['reference_indices']
        
        # Sort by ground truth distance if configured
        if STARCv2Config.SORT_BY_GT_DISTANCE:
            # Create sorting indices (exclude reference rewards from sorting)
            non_ref_indices = [i for i in range(len(names)) 
                             if i not in ref_indices.values()]
            ref_only_indices = list(ref_indices.values())
            
            # Sort non-reference rewards by GT distance
            sorted_non_ref = sorted(non_ref_indices, key=lambda i: gt_distances[i])
            
            # Combine: references first, then sorted rewards
            sort_order = ref_only_indices + sorted_non_ref
        else:
            sort_order = list(range(len(names)))
        
        precision = STARCv2Config.MATRIX_PRECISION
        
        lines = []
        lines.append("STARC DISTANCE MATRIX (Compact Format)")
        lines.append("=" * 60)
        lines.append("")
        
        # Summary statistics
        lines.append("SUMMARY:")
        lines.append(f"Total rewards analyzed: {len(names)}")
        lines.append(f"Generated rewards: {len(names) - 2}")  # Exclude GT and -GT
        lines.append(f"Reference rewards: GROUND_TRUTH, NEGATIVE_GROUND_TRUTH")
        lines.append("")
        
        # Distance to references for all generated rewards
        non_ref_indices = [i for i, orig_i in enumerate(sort_order) 
                          if orig_i not in ref_indices.values()]
        
        if non_ref_indices:
            lines.append("DISTANCES TO REFERENCE BOUNDS:")
            lines.append(f"{'Reward Name':<25} {'→ GT':<10} {'→ -GT':<10}")
            lines.append("-" * 45)
            
            for i in non_ref_indices:
                orig_idx = sort_order[i]
                name = names[orig_idx]
                gt_dist = gt_distances[orig_idx]
                neg_gt_dist = neg_gt_distances[orig_idx]
                lines.append(f"{name:<25} {gt_dist:<10.{precision}f} {neg_gt_dist:<10.{precision}f}")
            
            lines.append("")
            
            # Quick statistics
            gt_dists_only = [gt_distances[sort_order[i]] for i in non_ref_indices]
            neg_gt_dists_only = [neg_gt_distances[sort_order[i]] for i in non_ref_indices]
            
            lines.append("REFERENCE DISTANCE STATISTICS:")
            lines.append(f"GT distances  - min: {min(gt_dists_only):.{precision}f}, "
                        f"max: {max(gt_dists_only):.{precision}f}, "
                        f"mean: {np.mean(gt_dists_only):.{precision}f}")
            lines.append(f"-GT distances - min: {min(neg_gt_dists_only):.{precision}f}, "
                        f"max: {max(neg_gt_dists_only):.{precision}f}, "
                        f"mean: {np.mean(neg_gt_dists_only):.{precision}f}")
            lines.append("")
        
        # Pairwise distances (upper triangle only)
        lines.append("PAIRWISE DISTANCES (Upper Triangle):")
        lines.append("Format: Reward1 ↔ Reward2: distance")
        lines.append("")
        
        # Reorder matrix according to sort_order
        sorted_names = [names[i] for i in sort_order]
        sorted_matrix = matrix[np.ix_(sort_order, sort_order)]
        
        # Output upper triangle only
        pair_count = 0
        for i in range(len(sorted_names)):
            for j in range(i+1, len(sorted_names)):
                distance = sorted_matrix[i, j]
                lines.append(f"{sorted_names[i]} ↔ {sorted_names[j]}: {distance:.{precision}f}")
                pair_count += 1
        
        lines.append("")
        lines.append(f"Total unique pairwise distances: {pair_count}")
        lines.append(f"(Symmetric matrix - each distance shown once)")
        
        return "\n".join(lines)
    
    def _format_full_matrix(self, results: Dict) -> str:
        """Original full matrix format (kept for compatibility)"""
        return self._build_matrix_string(
            results['reward_names'], 
            results['distance_matrix'],
            results['ground_truth_distances'],
            results['negative_ground_distances'],
            results['reference_indices'],
            list(range(len(results['reward_names'])))
        )
    
    def _build_transition_batch(self, env, n=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build a batch of transitions for STARC analysis, including x_velocity."""
        if n is None:
            n = self.n_samples
        
        s_list, a_list, sp_list, xvel_list = [], [], [], []
        # Handle variable return values from reset()
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result
        
        # Track x_position for velocity calculation
        if hasattr(env, 'data'):
            prev_x_position = env.data.qpos[0]
        else:
            prev_x_position = 0.0
        
        for i in range(n):
            a = env.action_space.sample()
            s = obs.copy()
            # Take step and compute x_velocity
            if hasattr(env, 'data'):
                x_position_before = env.data.qpos[0]
            else:
                x_position_before = 0.0
            step_result = env.step(a)
            if len(step_result) == 5:
                obs, _, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                obs, _, done, _ = step_result
            if hasattr(env, 'data'):
                x_position_after = env.data.qpos[0]
            else:
                x_position_after = 0.0
            x_velocity = (x_position_after - x_position_before) / env.dt if hasattr(env, 'dt') else 0.0
            sp = obs.copy()
            
            s_list.append(s)
            a_list.append(a)
            sp_list.append(sp)
            xvel_list.append(x_velocity)
            
            # Reset periodically to avoid getting stuck
            if i % 100 == 0:
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    obs = reset_result[0]
                else:
                    obs = reset_result
                if hasattr(env, 'data'):
                    prev_x_position = env.data.qpos[0]
                else:
                    prev_x_position = 0.0
        
        return np.vstack(s_list), np.vstack(a_list), np.vstack(sp_list), np.array(xvel_list)
    
    def _process_rewards_with_references(self, reward_files: List[pathlib.Path], S: np.ndarray, A: np.ndarray, SP: np.ndarray, X_VEL: np.ndarray) -> Tuple[List[np.ndarray], List[str], Dict]:
        """Process reward functions including reference rewards (GT and -GT)"""
        vectors = []
        names = []
        
        # Process rewards (references + generated) via workers to fully utilize multi-GPU
        print("    📏 Processing reference rewards...")
        print(f"    📊 Processing {len(reward_files)} generated rewards...")

        from ..config import STARCv2Config
        n_workers = STARCv2Config.N_WORKERS or 0

        if n_workers > 0:
            import multiprocessing as mp
            # Avoid over-subscribing threads inside workers
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            ctx = mp.get_context("spawn")

            # Decide device mapping per worker
            # Respect explicit override if provided. Otherwise, distribute across available CUDA devices.
            override = os.getenv("STARC_SARSA_DEVICE")
            device_list: List[str]
            if override:
                # Interpret override smartly to enable multi-GPU fanout
                if override == "cpu":
                    device_list = ["cpu"]
                elif override == "cuda":
                    num_cuda = torch.cuda.device_count()
                    device_list = [f"cuda:{i}" for i in range(num_cuda)] if num_cuda > 0 else ["cpu"]
                elif override.startswith("cuda:"):
                    device_list = [override]
                else:
                    # Fallback: treat as a single device spec
                    device_list = [override]
            else:
                num_cuda = torch.cuda.device_count()
                if num_cuda > 0:
                    device_list = [f"cuda:{i}" for i in range(num_cuda)]
                else:
                    device_list = ["cpu"]

            # Build worker args for references first, then generated files
            worker_args = []
            all_items = ["__REF__:GT", "__REF__:NEG"] + [str(p) for p in reward_files]
            for idx, item in enumerate(all_items):
                worker_args.append(
                    (
                        item,
                        float(self.discount),
                        int(STARCv2Config.N_EPISODES_SARSA),
                        device_list[idx % len(device_list)],
                        S, A, SP, X_VEL,
                    )
                )

            with ctx.Pool(processes=n_workers) as pool:
                total = len(worker_args)
                processed = 0
                for name, vec, err in pool.imap_unordered(_process_reward_worker, worker_args, chunksize=1):
                    if err is None:
                        vectors.append(vec)
                        names.append(name)
                        processed += 1
                        print(f"      ✅ Processed {name}  ({processed}/{total})")
                        if os.getenv("STARC_SARSA_PROGRESS") == "1" and torch.cuda.is_available():
                            # Lightweight CUDA mem usage snapshot
                            try:
                                import torch as _torch
                                devs = list(range(_torch.cuda.device_count()))
                                mems = [int(_torch.cuda.memory_allocated(d)/1e6) for d in devs]
                                print(f"         CUDA mem (MB): {mems}")
                            except Exception:
                                pass
                    else:
                        processed += 1
                        print(f"      ❌ Failed to process {name}: {err}  ({processed}/{total})")
        else:
            # Choose single-device run for sequential path
            override = os.getenv("STARC_SARSA_DEVICE")
            if override:
                if override == "cuda":
                    device_single = "cuda:0" if torch.cuda.is_available() else "cpu"
                else:
                    device_single = override
            else:
                device_single = "cuda:0" if torch.cuda.is_available() else "cpu"
            for item in ["__REF__:GT", "__REF__:NEG"] + [str(p) for p in reward_files]:
                name, vec, err = _process_reward_worker((item, float(self.discount), int(STARCv2Config.N_EPISODES_SARSA), device_single, S, A, SP, X_VEL))
                if err is None:
                    vectors.append(vec)
                    names.append(name)
                    print(f"      ✅ Processed {name}")
                else:
                    print(f"      ❌ Failed to process {name}: {err}")
        
        # Locate reference indices by name after parallel processing
        try:
            gt_index = names.index("GROUND_TRUTH")
        except ValueError:
            gt_index = -1
        try:
            neg_gt_index = names.index("NEGATIVE_GROUND_TRUTH")
        except ValueError:
            neg_gt_index = -1
        reference_indices = {'ground_truth': gt_index, 'negative_ground_truth': neg_gt_index}
        
        print(f"  ✅ Successfully processed {len(vectors)-2}/{len(reward_files)} generated rewards + 2 references")
        return vectors, names, reference_indices

    def _process_rewards_only(self, reward_files: List[pathlib.Path], S: np.ndarray, A: np.ndarray, SP: np.ndarray, X_VEL: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
        """Process only the provided reward functions (no references)."""
        vectors: List[np.ndarray] = []
        names: List[str] = []

        from ..config import STARCv2Config
        n_workers = STARCv2Config.N_WORKERS or 0

        if n_workers > 0:
            import multiprocessing as mp
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            ctx = mp.get_context("spawn")

            override = os.getenv("STARC_SARSA_DEVICE")
            if override:
                if override == "cpu":
                    device_list = ["cpu"]
                elif override == "cuda":
                    num_cuda = torch.cuda.device_count()
                    device_list = [f"cuda:{i}" for i in range(num_cuda)] if num_cuda > 0 else ["cpu"]
                elif override.startswith("cuda:"):
                    device_list = [override]
                else:
                    device_list = [override]
            else:
                num_cuda = torch.cuda.device_count()
                device_list = [f"cuda:{i}" for i in range(num_cuda)] if num_cuda > 0 else ["cpu"]

            worker_args = []
            for idx, item in enumerate([str(p) for p in reward_files]):
                worker_args.append((item, float(self.discount), int(STARCv2Config.N_EPISODES_SARSA), device_list[idx % len(device_list)], S, A, SP, X_VEL))

            with ctx.Pool(processes=n_workers) as pool:
                total = len(worker_args)
                processed = 0
                for name, vec, err in pool.imap_unordered(_process_reward_worker, worker_args, chunksize=1):
                    if err is None:
                        vectors.append(vec)
                        names.append(name)
                        processed += 1
                        print(f"      ✅ Processed {name}  ({processed}/{total})")
                    else:
                        processed += 1
                        print(f"      ❌ Failed to process {name}: {err}  ({processed}/{total})")
        else:
            override = os.getenv("STARC_SARSA_DEVICE")
            if override:
                if override == "cuda":
                    device_single = "cuda:0" if torch.cuda.is_available() else "cpu"
                else:
                    device_single = override
            else:
                device_single = "cuda:0" if torch.cuda.is_available() else "cpu"
            for item in [str(p) for p in reward_files]:
                name, vec, err = _process_reward_worker((item, float(self.discount), int(STARCv2Config.N_EPISODES_SARSA), device_single, S, A, SP, X_VEL))
                if err is None:
                    vectors.append(vec)
                    names.append(name)
                    print(f"      ✅ Processed {name}")
                else:
                    print(f"      ❌ Failed to process {name}: {err}")

        print(f"  ✅ Successfully processed {len(vectors)}/{len(reward_files)} rewards (no references)")
        return vectors, names

    # -------------------- Helper methods (class scope) --------------------
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
        # Ensure LLM-generated rewards can subclass RewardFunc without explicit import
        try:
            from starc.core.reward_func import RewardFunc as _BaseRewardFunc  # type: ignore
            setattr(module, "RewardFunc", _BaseRewardFunc)
        except Exception:
            pass
        spec.loader.exec_module(module)
        # Find the reward class dynamically
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and 
                hasattr(obj, '__bases__') and 
                any('RewardFunc' in str(base) for base in obj.__bases__)):
                return obj()
        # Fallback naming patterns
        possible_names = [
            "RewardFunc",
            f"RewardFunc_{path.stem.split('_')[-1]}",
            f"RewardFunc{path.stem.split('_')[-1]}",
        ]
        for name in possible_names:
            if hasattr(module, name):
                return getattr(module, name)()
        raise ImportError(f"Could not find a valid RewardFunc class in {path}")

    def _compute_reward_vector(self, env, S: np.ndarray, A: np.ndarray, SP: np.ndarray, X_VEL: np.ndarray) -> np.ndarray:
        """Compute canonicalized and normalized reward vector using StateVals when available.

        Canonical form (VAL-2): r_canon = r(s,a,s') - V(s) + discount * V(s')
        """
        import inspect
        # Determine if reward expects x_velocity
        if hasattr(env.reward_func, '__call__'):
            sig = inspect.signature(env.reward_func.__call__)
        else:
            sig = inspect.signature(env.reward_func)
        param_names = list(sig.parameters.keys())
        expects_xvel = 'x_velocity' in param_names

        # Vectorized raw reward over fixed transitions
        raw_f = self._canon_callable(env.reward_func_curried, expects_xvel)
        r_vec = raw_f(S, A, SP, X_VEL) if expects_xvel else raw_f(S, A, SP)
        if r_vec is None:
            raise ValueError("Canon function returned None")
        r_vec = r_vec.astype(np.float32)

        # StateVals-based canonicalization if available
        V = getattr(env, 'state_vals', None)
        if V is not None:
            try:
                # Prefer batch evaluation when available
                if hasattr(V, 'evaluate_batch'):
                    vs = V.evaluate_batch(S)
                    vsp = V.evaluate_batch(SP)
                else:
                    vs = np.array([V(s) for s in S], dtype=np.float32)
                    vsp = np.array([V(sp) for sp in SP], dtype=np.float32)
                canon_vec = r_vec - vs + (self.discount * vsp)
            except Exception as e:
                print(f"    ⚠️  StateVals canonicalization failed ({e}); falling back to raw reward vector.")
                canon_vec = r_vec
        else:
            canon_vec = r_vec

        # Sanitize NaNs/Infs before normalization
        canon_vec = np.nan_to_num(canon_vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # L2 normalize canonical vector
        norm = float(np.linalg.norm(canon_vec))
        if norm > 1e-8:
            canon_vec = canon_vec / norm
        return canon_vec.astype(np.float32)

    def _canon_callable(self, reward_callable, expects_xvel=False):
        """Create a canonical callable for the reward function, supporting x_velocity if needed."""
        def vectorized_reward(S, A, SP, X_VEL=None):
            results = []
            for i in range(len(S)):
                try:
                    # Some LLM-generated rewards index state[12:18] assuming 18-dim obs.
                    # Our obs is 17-dim (8 pos w/o x + 9 vel). Pad by 1 to make 18 for safe slicing.
                    s_i = S[i]
                    sp_i = SP[i]
                    if isinstance(s_i, np.ndarray) and s_i.shape[0] == 17:
                        s_i = np.pad(s_i, (0, 1), mode='constant')
                    if isinstance(sp_i, np.ndarray) and sp_i.shape[0] == 17:
                        sp_i = np.pad(sp_i, (0, 1), mode='constant')

                    if expects_xvel:
                        r = reward_callable(s_i, A[i], sp_i, X_VEL[i])
                    else:
                        r = reward_callable(s_i, A[i], sp_i)
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
        distance_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = np.nan_to_num(vectors[i] - vectors[j], nan=0.0, posinf=0.0, neginf=0.0)
                    distance_matrix[i, j] = float(np.linalg.norm(diff))
        # Ensure finite matrix
        distance_matrix = np.nan_to_num(distance_matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return distance_matrix


# Module-level worker to be picklable by multiprocessing
def _process_reward_worker(args):
    import pathlib
    from .half_cheetah_env_fixed import HalfCheetahEnvFixed as HalfCheetahEnv
    try:
        (path_str, discount, n_episodes_sarsa, device_str, S, A, SP, X_VEL) = args
        path = pathlib.Path(path_str) if not str(path_str).startswith("__REF__:") else None
        # Set device override for SARSA/StateVals in this worker
        if device_str:
            os.environ["STARC_SARSA_DEVICE"] = device_str
            # Explicitly set device for torch in this process for isolation
            try:
                if device_str.startswith("cuda"):
                    import torch as _torch
                    if ":" in device_str:
                        idx = int(device_str.split(":")[1])
                    else:
                        idx = 0
                    if _torch.cuda.is_available():
                        _torch.cuda.set_device(idx)
            except Exception:
                pass
        # Local import to access analyzer helpers
        analyzer = STARCv2Analyzer()
        # Handle reference rewards
        if path is None:
            tag = str(path_str)
            from ..config import STARCv2Config
            env_name = STARCv2Config.ENV_NAME
            if tag == "__REF__:GT":
                if env_name == 'ant':
                    reward_obj = GroundTruthAntReward()
                elif env_name == 'humanoid':
                    reward_obj = GroundTruthHumanoidReward()
                else:
                    reward_obj = GroundTruthReward()
                name = "GROUND_TRUTH"
            else:
                if env_name == 'ant':
                    reward_obj = NegativeGroundAntReward()
                elif env_name == 'humanoid':
                    reward_obj = NegativeGroundHumanoidReward()
                else:
                    reward_obj = NegativeGroundReward()
                name = "NEGATIVE_GROUND_TRUTH"
        else:
            reward_obj = analyzer._load_reward_from_file(path)
            name = path.stem
        env = HalfCheetahEnv(reward_obj, discount, n_episodes_sarsa)
        vector = analyzer._compute_reward_vector(env, S, A, SP, X_VEL)
        return (name, vector, None)
    except Exception as e:
        import traceback
        return (str(path_str), None, f"{e}\n{traceback.format_exc()}")