#!/usr/bin/env python3
"""Cluster all reward functions in *rewards/T1* using STARC distance.

The script collects every ``rewardfunc_*.py`` file from **iteration_1** …
**iteration_3** (16 per iteration) and groups them into clusters based on the
STARC pair-wise distance and the configured threshold in ``config.py``.

Output files are written to ``results/T1/``:

    • ``clusters.json`` – mapping *cluster_id* → list of reward names
    • ``clusters.txt``  – human-readable text summary

Run from the **project root** (``starc_v2``):

```bash
python -m analysis.cluster_T1_rewards            # default threshold
# or override:
python -m analysis.cluster_T1_rewards --threshold 0.10
```
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
from typing import List, Dict

import numpy as np

from starc_v2.core.starc_analyzer import STARCv2Analyzer
from starc_v2.pipeline import STARCv2Pipeline  # for the static _cluster_rewards helper
from starc_v2.config import STARCv2Config

# ---------------------------------------------------------------------------
# Helper --------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _find_reward_files(t1_dir: pathlib.Path) -> List[pathlib.Path]:
    """Return a sorted list of *all* rewardfunc_*.py files under T1 iterations."""
    reward_files: List[pathlib.Path] = []
    for iteration_dir in (t1_dir / "iteration_1", t1_dir / "iteration_2", t1_dir / "iteration_3"):
        if iteration_dir.exists():
            reward_files.extend(sorted(iteration_dir.rglob("rewardfunc_*.py")))
    return reward_files


# ---------------------------------------------------------------------------
# Main ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():  # noqa: D401
    parser = argparse.ArgumentParser(description="Cluster STARC v2 T1 reward functions")
    parser.add_argument("--threshold", type=float, default=STARCv2Config.CLUSTER_THRESHOLD,
                        help="Distance threshold for clustering (default: value from config.py)")
    parser.add_argument("--seed", type=int, help="Random seed for selecting GA seed rewards (optional)")
    parser.add_argument("--target-count", type=int, default=30,
                        help="Total number of seed rewards to output (default: 30)")
    args = parser.parse_args()

    import random  # local import to keep stdlib first
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    root = pathlib.Path(__file__).resolve().parents[1]
    t1_dir = root / "rewards" / "T1"
    if not t1_dir.exists():
        raise FileNotFoundError(t1_dir)

    reward_files = _find_reward_files(t1_dir)
    if len(reward_files) == 0:
        raise RuntimeError("No rewardfunc_*.py files found in rewards/T1/*")

    print(f"🔍 Found {len(reward_files)} reward files from T1 iterations")

    # ------------------------------------------------------------------
    print("🔬 Computing STARC distance matrix … (this can take a minute)")
    analyzer = STARCv2Analyzer()

    out_dir = root / "results" / "T1"
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_file = out_dir / "transition_batch.npz"

    if batch_file.exists():
        print("📥 Loading cached transition batch …")
        cache = np.load(batch_file)
        precomputed = (cache["S"], cache["A"], cache["SP"], cache["X_VEL"])
        results = analyzer.analyze_rewards(reward_files, iteration=0, precomputed_batch=precomputed)
    else:
        print("🛠  Building transition batch (first run) …")
        results = analyzer.analyze_rewards(reward_files, iteration=0)
        # Save for future determinism
        np.savez(batch_file, **results["transition_data"])
        print(f"💾 Cached transition batch → {batch_file}")

    # ------------------------------------------------------------------
    print(f"🔗 Clustering with threshold = {args.threshold:.3f}")
    clusters: Dict[str, List[str]] = STARCv2Pipeline._cluster_rewards(
        results["distance_matrix"],
        results["reward_names"],
        args.threshold,
    )

    # ------------------------------------------------------------------
    # Save clustering results ------------------------------------------
    out_dir = root / "results" / "T1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clusters.json").write_text(json.dumps(clusters, indent=2))

    # Pretty text output ------------------------------------------------
    lines = []
    for cid, members in clusters.items():
        lines.append(f"{cid}: {', '.join(members)}")
    (out_dir / "clusters.txt").write_text("\n".join(lines))

    print(f"✅ Saved clusters → {out_dir}/clusters.json  (k={len(clusters)})")

    # ------------------------------------------------------------------
    # Derive GA seed list ----------------------------------------------
    # Map reward name → file path for quick lookup
    name_to_path = {p.stem: p for p in reward_files}

    # 1st pass: mandatory picks per cluster
    selected = []
    remaining_pool = []  # tuples (cluster_id, reward_name)

    for cid, members in clusters.items():
        mlist = members.copy()
        random.shuffle(mlist)
        if len(mlist) == 1:
            selected.append(mlist[0])
        elif len(mlist) == 2:
            selected.extend(mlist)  # pick both (already shuffled)
        else:
            # pick two random representatives
            selected.extend(mlist[:2])
            for rem in mlist[2:]:
                remaining_pool.append((cid, rem))

    # 2nd pass: top-up until reaching target-count (or exhaust pool)
    random.shuffle(remaining_pool)
    idx = 0
    while len(selected) < args.target_count and idx < len(remaining_pool):
        selected.append(remaining_pool[idx][1])
        idx += 1

    print(f"🚀 Prepared {len(selected)} rewards to seed GA")
    if len(selected) < args.target_count:
        print(f"⚠️  Warning: total selected rewards < {args.target_count}; GA population will be smaller")

    # Save selection paths ------------------------------------------------
    ga_list_txt = []
    ga_list = []
    for name in selected:
        path = str(name_to_path[name].relative_to(root)) if name in name_to_path else name
        ga_list_txt.append(path)
        ga_list.append(path)

    (out_dir / "ga_seed_rewards.txt").write_text("\n".join(ga_list_txt))
    (out_dir / "ga_seed_rewards.json").write_text(json.dumps(ga_list, indent=2))
    print(f"💾 GA seed list saved → {out_dir}/ga_seed_rewards.txt")


if __name__ == "__main__":
    main() 