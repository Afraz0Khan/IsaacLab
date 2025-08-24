#!/usr/bin/env python3
"""
Select a fixed number of diverse reward seeds from STARC v2 final clusters
and write them to the file BO expects: starc_v2/results/T1/ga_seed_rewards.json

Rules:
  1) Pick at least one reward from each cluster
  2) Within a cluster, pick rewards that are far apart (farthest-point sampling)
  3) Large clusters get at least 3–4 seeds if needed to reach the target count

By default, the script reads:
  - results/final_clusters.json
  - the latest iteration_* folder's distance_matrix.npy and reward_names.json
  - scans rewards/* for matching reward files

Usage (from repo root):
  python -m starc_v2.scripts.select_seeds_from_clusters \
    --target 30 \
    --large-threshold 6 --large-min 4

You can also override paths if needed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def find_latest_iteration(results_dir: Path) -> Path:
    iters = sorted(
        (p for p in results_dir.glob("iteration_*") if p.is_dir()),
        key=lambda p: int(re.search(r"iteration_(\d+)", p.name).group(1)),
    )
    if not iters:
        raise RuntimeError(f"No iteration_* folders under {results_dir}")
    return iters[-1]


def farthest_sampling(items: List[str], idx_lookup: Dict[str, int], D: np.ndarray, k: int) -> List[str]:
    if not items or k <= 0:
        return []
    if k >= len(items):
        return list(items)
    idxs = [idx_lookup[n] for n in items]
    subD = D[np.ix_(idxs, idxs)]
    first = items[int(subD.mean(axis=1).argmax())]
    picked = [first]
    while len(picked) < k:
        best, cand = -1.0, None
        for n in items:
            if n in picked:
                continue
            ndx = idx_lookup[n]
            md = min(D[ndx, idx_lookup[p]] for p in picked)
            if md > best:
                best, cand = md, n
        if cand is None:
            break
        picked.append(cand)
    return picked


def resolve_reward_paths(names: List[str], rewards_root: Path) -> List[str]:
    paths: List[str] = []
    for n in names:
        matches = sorted(
            rewards_root.glob(f"**/{n}.py"),
            key=lambda p: int(re.search(r"iteration_(\d+)", str(p)).group(1))
            if re.search(r"iteration_(\d+)", str(p))
            else -1,
        )
        if matches:
            paths.append(str(matches[-1].resolve()))
            continue
        # fallback: any stem match
        any_match = list(rewards_root.glob(f"**/{n}*.py"))
        if any_match:
            paths.append(str(any_match[-1].resolve()))
    return paths


def main():
    parser = argparse.ArgumentParser(description="Select diverse STARC seeds from final clusters")
    parser.add_argument("--base", default=str(Path(__file__).resolve().parents[1]),
                        help="Base path to starc_v2 (default: repository starc_v2)")
    parser.add_argument("--results", default=None,
                        help="Path to results dir (default: <base>/results)")
    parser.add_argument("--rewards", default=None,
                        help="Path to rewards dir (default: <base>/rewards)")
    parser.add_argument("--clusters", default=None,
                        help="Path to final_clusters.json (default: <results>/final_clusters.json)")
    parser.add_argument("--target", type=int, default=30,
                        help="Total number of seeds to select (default: 30)")
    parser.add_argument("--large-threshold", type=int, default=6,
                        help="Cluster size threshold to be considered large (default: 6)")
    parser.add_argument("--large-min", type=int, default=4,
                        help="Minimum picks for large clusters (default: 4)")
    parser.add_argument("--outdir", default=None,
                        help="Output dir (default: <results>/T1)")

    args = parser.parse_args()

    base = Path(args.base)
    results_dir = Path(args.results) if args.results else (base / "results")
    rewards_root = Path(args.rewards) if args.rewards else (base / "rewards")
    clusters_path = Path(args.clusters) if args.clusters else (results_dir / "final_clusters.json")
    out_dir = Path(args.outdir) if args.outdir else (results_dir / "T1")

    latest = find_latest_iteration(results_dir)
    dm_path = latest / "distance_matrix.npy"
    names_path = latest / "reward_names.json"

    D = np.load(dm_path)
    names = json.loads(names_path.read_text())
    # Exclude references
    names = [n for n in names if n not in ("GROUND_TRUTH", "NEGATIVE_GROUND_TRUTH")]
    name_to_idx = {n: i for i, n in enumerate(names)}

    clusters: Dict[str, List[str]] = json.loads(clusters_path.read_text())
    cluster_names = {k: [n for n in v if n in name_to_idx] for k, v in clusters.items()}
    cluster_names = {k: v for k, v in cluster_names.items() if v}

    target_total = max(1, args.target)

    # Initial allocation: at least 1 per cluster
    counts = {k: 1 for k in cluster_names}
    # Large clusters get a higher floor
    for k, items in cluster_names.items():
        if len(items) >= args.large_threshold:
            counts[k] = min(args.large_min, len(items))

    def total():
        return sum(counts.values())

    # Reduce if over target
    if total() > target_total:
        for k in sorted(cluster_names, key=lambda x: len(cluster_names[x]), reverse=True):
            while counts[k] > 1 and total() > target_total:
                counts[k] -= 1

    # Increase if under target (favor larger clusters)
    while total() < target_total:
        progressed = False
        for k in sorted(cluster_names, key=lambda x: len(cluster_names[x]), reverse=True):
            if counts[k] < len(cluster_names[k]) and total() < target_total:
                counts[k] += 1
                progressed = True
            if total() >= target_total:
                break
        if not progressed:
            break

    # Pick per cluster using farthest sampling
    selected: List[str] = []
    for k, items in cluster_names.items():
        selected += farthest_sampling(items, name_to_idx, D, counts[k])

    selected = selected[:target_total]
    paths = resolve_reward_paths(selected, rewards_root)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ga_seed_rewards.json").write_text(json.dumps(paths, indent=2))
    (out_dir / "ga_seed_rewards.txt").write_text("\n".join(paths))
    print(f"Wrote {len(paths)} seeds to {out_dir/'ga_seed_rewards.json'}")


if __name__ == "__main__":
    sys.exit(main())

