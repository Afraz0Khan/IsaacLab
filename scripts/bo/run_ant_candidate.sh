#!/usr/bin/env bash
set -euo pipefail

# ----- CONFIG -----
TASK="Isaac-Ant-v0"
NUM_ENVS=128
HORIZON=16
MINIBATCH=2048
MAX_ITERS=147           # ~300k frames
EVAL_STEPS=2000
EVAL_ENVS=16
SEEDS=(0 1 2 3 4 5 6 7 8 9 10)
HEADLESS="--headless"

# Reward override example (Hydra overrides)
REWARD_OVERRIDES=(
  # e.g.: 'env.reward_cfg.move_to_target_w=0.5'
  # e.g.: 'env.reward_cfg.energy_w=-0.05'
)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
LOG_BASE="${HOME}/Documents/IsaacLab/logs/rl_games/ant"

RESULTS=()

for SEED in "${SEEDS[@]}"; do
  echo ">>> SEED ${SEED}"

  # 1) Train with fixed budget
  "${ROOT_DIR}/isaaclab.sh" -p scripts/reinforcement_learning/rl_games/train.py \
    --task "${TASK}" ${HEADLESS} \
    --num_envs "${NUM_ENVS}" \
    agent.params.config.horizon_length="${HORIZON}" \
    agent.params.config.minibatch_size="${MINIBATCH}" \
    --max_iterations "${MAX_ITERS}" \
    --seed "${SEED}" \
    "${REWARD_OVERRIDES[@]}"

  # 2) Resolve last checkpoint under the most recent run
  CKPT=$(ls -1t "${LOG_BASE}"/*/nn/last_*.pth | head -n1)
  if [[ -z "${CKPT:-}" ]]; then
    echo "ERROR: Could not find last_*.pth under ${LOG_BASE}/*/nn/" >&2
    exit 1
  fi
  echo "Found checkpoint: ${CKPT}"

  # 3) Deterministic eval -> prints a single float
  DIST=$("${ROOT_DIR}/isaaclab.sh" -p scripts/evaluation/eval_distance.py \
    --task "${TASK}" ${HEADLESS} \
    --checkpoint "${CKPT}" \
    --eval_steps "${EVAL_STEPS}" \
    --eval_num_envs "${EVAL_ENVS}" \
    --seed "${SEED}" | tail -n1)
  echo "seed ${SEED} distance: ${DIST}"
  RESULTS+=("${DIST}")
done

# 4) Aggregate (mean, std)
{
  for v in "${RESULTS[@]}"; do echo "$v"; done
} | python - "$@" <<'PY'
import sys
import numpy as np
vals = []
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        vals.append(float(line))
    except Exception:
        pass
if not vals:
    print("BO_FITNESS_MEAN 0.0")
    print("BO_FITNESS_STD 0.0")
    sys.exit(0)
print("BO_FITNESS_MEAN", np.mean(vals))
print("BO_FITNESS_STD", np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
PY


