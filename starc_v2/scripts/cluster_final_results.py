#!/usr/bin/env python3
"""
Cluster STARC results exactly like the end of the pipeline and write final_clusters.json.

Usage (from repo root):
  python -m starc_v2.scripts.cluster_final_results --env halfcheetah

This mirrors STARCv2Pipeline._cluster_rewards on the latest iteration's matrix.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from starc_v2.config import STARCv2Config
from starc_v2.pipeline import STARCv2Pipeline


def _find_latest_iteration(results_dir: Path) -> Path:
    def extract_iter_num(p: Path) -> int:
        match = re.search(r"iteration_(\d+)", p.name)
        return int(match.group(1)) if match else 0
    
    iters = sorted(
        (p for p in results_dir.glob("iteration_*") if p.is_dir()),
        key=extract_iter_num,
    )
    if not iters:
        raise RuntimeError(f"No iteration_* folders under {results_dir}")
    return iters[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Cluster STARC results like the pipeline and write final_clusters.json"
    )
    parser.add_argument("--env", choices=["halfcheetah", "ant", "humanoid"], default=None,
                        help="Environment name (overrides STARCv2Config.ENV_NAME)")
    parser.add_argument("--results", default=None,
                        help="Results base dir (default: starc_v2/results/<env>)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override clustering threshold (else use config)")
    args = parser.parse_args()
    
    # Type annotations for clarity
    env_name = args.env
    results_path = args.results
    threshold_override = args.threshold

    # Resolve env and paths
    if env_name:
        STARCv2Config.ENV_NAME = env_name
    env = STARCv2Config.ENV_NAME

    base = Path(__file__).resolve().parents[1]
    results_base = Path(results_path) if results_path else (base / "results" / env)

    latest = _find_latest_iteration(results_base)
    dm_path = latest / "distance_matrix.npy"
    names_path = latest / "reward_names.json"

    if not dm_path.exists() or not names_path.exists():
        raise FileNotFoundError(f"Missing inputs: {dm_path} or {names_path}")

    D = np.load(dm_path)
    with open(names_path, "r") as f:
        names: List[str] = json.load(f)

    # Use the exact clustering helper from the pipeline
    threshold = threshold_override if threshold_override is not None else STARCv2Config.CLUSTER_THRESHOLD

    # Ensure numeric stability
    D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)

    clusters: Dict[str, List[str]] = STARCv2Pipeline._cluster_rewards(D, names, threshold)  # type: ignore

    out_path = results_base / "final_clusters.json"
    with open(out_path, "w") as f:
        json.dump(clusters, f, indent=2)

    print(f"Wrote clusters → {out_path}")


if __name__ == "__main__":
    main()


