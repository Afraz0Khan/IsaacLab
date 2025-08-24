#!/usr/bin/env python3
"""
Cleanup script for STARC v2 reward-evolution pipeline.

This utility deletes artefacts produced during pipeline runs so that you can
re-run the pipeline from a clean slate.  It performs **three** actions:

1. Removes generated reward files located in  ``starc_v2/rewards/iteration_*``
   (core rewards such as ``GroundTruthReward`` are preserved).
2. Removes SARSA value-function models for numbered rewards stored in
   ``starc_v2/sarsa_models/RewardFunc_*`` (core models are preserved).
3. Deletes the pipeline result files inside ``starc_v2/results``.

The script purposefully leaves the following untouched:
• Core reward implementations (e.g. *GroundTruthReward*, *NegativeGroundRewardFixed*).
• Core SARSA models for those base rewards.
• Prompt templates inside ``starc_v2/llm/prompts``.

Run from the project root:

```
python -m starc_v2.cleanup_starcv2  # or `python starc_v2/cleanup_starcv2.py`
```
"""

import re
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _base_dir() -> Path:
    """Return absolute path to the *starc_v2* package directory."""
    return Path(__file__).resolve().parent


def _delete_dir(path: Path):
    """Delete directory *path* if it exists (recursively)."""
    if path.exists():
        shutil.rmtree(path)
        print(f"  ✅ Deleted {path.relative_to(_base_dir())}/")
    else:
        print(f"  ℹ️  {path.relative_to(_base_dir())}/ does not exist – skipped")

# ---------------------------------------------------------------------------
# Cleanup steps
# ---------------------------------------------------------------------------

def clean_generated_rewards():
    """Remove *iteration_* sub-directories under *rewards/*."""
    rewards_dir = _base_dir() / "rewards"
    if not rewards_dir.exists():
        print("Rewards directory missing – nothing to clean")
        return

    print("🗑️  Cleaning generated rewards …")
    for sub in rewards_dir.iterdir():
        if sub.is_dir() and re.match(r"iteration_\d+", sub.name):
            _delete_dir(sub)
        elif sub.name == "__pycache__":
            _delete_dir(sub)
    print("Finished cleaning rewards\n")


def clean_sarsa_models():
    """Remove SARSA models for numbered reward functions."""
    sarsa_dir = _base_dir() / "sarsa_models"
    if not sarsa_dir.exists():
        print("SARSA models directory missing – nothing to clean")
        return

    print("🗑️  Cleaning SARSA models …")
    for sub in sarsa_dir.iterdir():
        if sub.is_dir() and re.match(r"RewardFunc_\d+", sub.name):
            _delete_dir(sub)
    print("Finished cleaning SARSA models\n")


def clean_results():
    """Delete the *results* directory entirely."""
    results_dir = _base_dir() / "results"
    print("🗑️  Cleaning pipeline results …")
    _delete_dir(results_dir)
    # Recreate empty results folder for future runs
    results_dir.mkdir(exist_ok=True)
    print("Created fresh results/ directory\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("STARC v2 – CLEANUP UTILITY")
    print("=" * 60)

    root = _base_dir()
    print(f"Working directory: {root}\n")

    # Short confirmation prompt
    resp = input(
        "This will DELETE generated reward files, SARSA models for RewardFunc_* and ALL pipeline results. Continue? (y/N): "
    )
    if resp.lower() != "y":
        print("Aborted – no files were removed.")
        return

    print()
    clean_generated_rewards()
    clean_sarsa_models()
    clean_results()

    print("=" * 60)
    print("CLEANUP COMPLETE – your workspace is ready for a new run ✨")
    print("=" * 60)


if __name__ == "__main__":
    main() 