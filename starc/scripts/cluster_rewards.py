# cluster_rewards.py
import importlib.util, pathlib, json
import numpy as np
from sklearn.cluster import DBSCAN
from functools import partial

from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.rewards.ground_truth_reward import GroundTruthReward

DISCOUNT  = 0.848
N_EPISODE = 6000                  # for StateVals / SARSA
N_SAMPLES = 96                    # every canonical vector length
CLUSTER_EPS = 0.15                # distance threshold

def canon_callable(reward_callable, env_info, *, n_canon=32, n_norm=96):
    """
    Simplified canonical function that just returns the raw reward callable
    vectorized, avoiding the circular import issue.
    """
    def vectorized_reward(S, A, SP):
        """Vectorized version that processes arrays of states/actions"""
        results = []
        for s, a, sp in zip(S, A, SP):
            r = reward_callable(s, a, sp)
            results.append(r)
        return np.array(results)
    
    return vectorized_reward

def load_reward_from_file(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)        # type: ignore
    
    # Find the reward class dynamically (should be the one that inherits from RewardFunc)
    for name in dir(module):
        obj = getattr(module, name)
        if (isinstance(obj, type) and 
            hasattr(obj, '__bases__') and 
            any('RewardFunc' in str(base) for base in obj.__bases__)):
            return obj()
    
    # Fallback: try the old method
    return getattr(module, "RewardFunc")()

# ------------------------------------------------------------------
def build_transition_batch(env, n=N_SAMPLES):
    s_list, a_list, sp_list = [], [], []
    obs, _ = env.reset()
    print(f"    🎯 Sampling {n} transitions...")
    
    for i in range(n):
        a = env.action_space.sample()
        s = obs.copy()
        obs, *_ = env.step(a)[:4]
        sp = obs.copy()
        s_list.append(s); a_list.append(a); sp_list.append(sp)
        
        # Progress indicator every 25% and reset every 100 steps
        if i % (n // 4) == 0 and i > 0:
            print(f"      📊 Progress: {i}/{n} ({100*i//n}%)")
        if i % 100 == 0:
            obs, _ = env.reset()
    
    print(f"    ✅ Completed {n} transitions")
    return np.vstack(s_list), np.vstack(a_list), np.vstack(sp_list)

# ------------------------------------------------------------------
def main(folder_with_py_rewards="starc/rewards/llm", out="clusters.json"):
    print(f"🔍 Looking for reward files in: {folder_with_py_rewards}")
    reward_paths = [p for p in pathlib.Path(folder_with_py_rewards).glob("*.py") if p.name != "__init__.py"]
    print(f"✅ Loaded {len(reward_paths)} reward files")
    
    if len(reward_paths) == 0:
        print("❌ No reward files found! Check the directory path.")
        return

    print(f"📁 Found files: {[p.name for p in reward_paths]}")

    # ---- build env just once for spaces and env_info template ----
    print(f"🏗️  Building dummy environment (DISCOUNT={DISCOUNT}, N_EPISODE={N_EPISODE})...")
    dummy_env = HalfCheetahEnv(
        GroundTruthReward(), DISCOUNT, N_EPISODE
    )
    print(f"🎲 Generating {N_SAMPLES} transition samples...")
    S, A, SP = build_transition_batch(dummy_env)   # shape (96,dim)
    print(f"✅ Generated transition batch: S{S.shape}, A{A.shape}, SP{SP.shape}")

    vectors, names = [], []
    print(f"\n🔄 Processing {len(reward_paths)} reward functions...")

    for i, p in enumerate(reward_paths, 1):
        print(f"  [{i}/{len(reward_paths)}] Processing {p.name}...")
        try:
            rew_obj  = load_reward_from_file(p)
            print(f"    ✅ Loaded reward class: {rew_obj.__class__.__name__}")
            
            env      = HalfCheetahEnv(rew_obj, DISCOUNT, N_EPISODE)
            print(f"    🏗️  Created environment...")
            
            canon_f  = canon_callable(env.reward_func_curried, env.env_info)
            print(f"    🔧 Created canonical function...")
            
            vec      = canon_f(S, A, SP).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-8
            print(f"    📊 Generated vector (norm: {np.linalg.norm(vec):.4f})")
            
            vectors.append(vec)
            names.append(p.stem)
            
        except Exception as e:
            print(f"    ❌ Failed to process {p.name}: {e}")
            continue

    if len(vectors) == 0:
        print("❌ No reward functions were successfully processed!")
        return

    print(f"\n🧮 Clustering {len(vectors)} reward vectors...")
    V = np.vstack(vectors)                         # (k, 96)
    print(f"✅ Stacked vectors: {V.shape}")
    
    # ---- DBSCAN on the pair‑wise L2 distance matrix ----
    print(f"📏 Computing pairwise distances...")
    dmat = np.linalg.norm(V[:, None] - V[None, :], axis=-1)
    print(f"✅ Distance matrix: {dmat.shape}")
    
    print(f"🎯 Running DBSCAN clustering (eps={CLUSTER_EPS})...")
    model = DBSCAN(eps=CLUSTER_EPS, metric="precomputed",
                   min_samples=1).fit(dmat)

    clusters = {}
    for idx, cid in enumerate(model.labels_):
        clusters.setdefault(int(cid), []).append(names[idx])

    print(f"\n🎉 Clustering complete! Found {len(clusters)} clusters:")
    for cluster_id, members in clusters.items():
        print(f"  Cluster {cluster_id}: {len(members)} members")

    print(f"\n📋 Detailed clustering results:")
    print(json.dumps(clusters, indent=2))
    
    print(f"\n💾 Saving results to {out}...")
    with open(out, "w") as f:
        json.dump(clusters, f, indent=2)
    print(f"✅ Results saved successfully!")

if __name__ == "__main__":
    main()
