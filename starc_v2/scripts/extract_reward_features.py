#!/usr/bin/env python3
"""
Scan reward functions and build a master dictionary of the features they declare via `get_features()`.

Input options (one of):
  - A directory containing rewards (recursively scans *.py)
  - A text file with one reward file path per line (e.g., ga_seed_rewards.txt)
  - A JSON file with a list of reward file paths (e.g., ga_seed_rewards.json)

Output:
  - Prints a summary table to stdout.
  - Saves `feature_catalog.json` alongside this script.

The resulting structure looks like:

```json
{
  "forward_velocity": {
    "expression": "x_velocity",
    "used_by": ["RewardFunc_1", "RewardFunc_7", ...]
  },
  "ctrl_cost": {
    "expression": "np.sum(np.square(action))",
    "used_by": ["RewardFunc_1", "RewardFunc_3", ...]
  }
}
```

Run from the repository root:

```bash
python starc_v2/scripts/extract_reward_features.py
```
"""

import json
import pathlib
import pprint
import importlib.util
import argparse
from typing import Dict, List, Iterable

from starc.core.reward_func import RewardFunc  # base class

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT_PATH = pathlib.Path(__file__).with_name("feature_catalog.json")


def load_reward_class(path: pathlib.Path):
    """Dynamically load a reward class from *path* and return an instance."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec.loader is not None
    # Ensure RewardFunc base is present for LLM-generated files that don't import it
    setattr(module, "RewardFunc", RewardFunc)
    spec.loader.exec_module(module)  # type: ignore
    # Find subclass of RewardFunc
    for attr in dir(module):
        obj = getattr(module, attr)
        if isinstance(obj, type) and issubclass(obj, RewardFunc) and obj is not RewardFunc:
            return obj()
    raise RuntimeError(f"No RewardFunc subclass found in {path}")


def iter_reward_files(input_path: pathlib.Path) -> Iterable[pathlib.Path]:
    """Yield reward file paths from a directory, txt list, or json list."""
    if input_path.is_dir():
        yield from (p for p in input_path.rglob("*.py") if not p.name.startswith("__"))
        return
    # Text list
    if input_path.suffix.lower() in {".txt"}:
        for line in input_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            p = pathlib.Path(line)
            if not p.is_absolute():
                p = (ROOT / p).resolve()
            yield p
        return
    # JSON list
    if input_path.suffix.lower() in {".json"}:
        data = json.loads(input_path.read_text())
        if isinstance(data, list):
            for s in data:
                p = pathlib.Path(s)
                if not p.is_absolute():
                    p = (ROOT / p).resolve()
                yield p
            return
    raise RuntimeError(f"Unsupported input: {input_path}")


def collect_features(input_path: pathlib.Path) -> Dict[str, Dict[str, List[str]]]:
    """Walk through provided rewards and aggregate features."""
    feature_map: Dict[str, Dict] = {}

    for py_file in iter_reward_files(input_path):
        try:
            reward_obj = load_reward_class(py_file)
            if not hasattr(reward_obj, "get_features"):
                continue  # skip if method missing
            features = reward_obj.get_features()
            if not isinstance(features, dict):
                continue
            reward_name = reward_obj.__class__.__name__
            for feat_name, expr in features.items():
                entry = feature_map.setdefault(feat_name, {"expression": expr, "used_by": []})
                # If we encountered the feature earlier but expression differs, keep first seen
                if reward_name not in entry["used_by"]:
                    entry["used_by"].append(reward_name)
        except Exception as e:
            print(f"⚠️  Failed to process {py_file.name}: {e}")
    return feature_map


def main():
    parser = argparse.ArgumentParser(description="Extract reward feature usage from reward files")
    parser.add_argument("input", nargs="?", default=str(ROOT / "results/T1/ga_seed_rewards.txt"),
                        help="Directory of rewards, or a txt/json list of reward paths (default: starc_v2/rewards)")
    parser.add_argument("--output", dest="output", default=str(OUTPUT_PATH),
                        help="Output path for feature_catalog.json (default: alongside script)")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    catalog = collect_features(input_path)

    print("\n📚 MASTER FEATURE CATALOG\n===========================")
    for feat, info in catalog.items():
        print(f"• {feat} -> {info['expression']} (used by {len(info['used_by'])} rewards)")
    print(f"\nTotal unique features: {len(catalog)}")

    out_path = pathlib.Path(args.output)
    with open(out_path, "w") as f:
        json.dump(catalog, f, indent=2)
    try:
        rel = out_path.relative_to(ROOT.parent)
    except Exception:
        rel = out_path
    print(f"\n💾 Saved catalog to {rel}")

    # ------------------------------------------------------------------
    # Save simplified dictionary: feature_name -> expression (first seen)
    # ------------------------------------------------------------------
    simple_dict = {feat: info["expression"] for feat, info in catalog.items()}

    simple_path = out_path.with_name("features_simple.json")
    with open(simple_path, "w") as f:
        json.dump(simple_dict, f, indent=2)
    try:
        rel2 = simple_path.relative_to(ROOT.parent)
    except Exception:
        rel2 = simple_path
    print(f"💾 Saved simplified feature map to {rel2}")


if __name__ == "__main__":
    main() 