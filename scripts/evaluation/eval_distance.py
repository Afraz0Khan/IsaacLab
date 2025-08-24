#!/usr/bin/env python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic evaluation: mean XY displacement over a fixed horizon for RL-Games checkpoints.

CLI:
  --task {Isaac-Ant-v0, Isaac-Humanoid-v0}
  --checkpoint /abs/path/to/last_*.pth
  --eval_steps 2000
  --eval_num_envs 16
  --seed 0
  --headless

Output:
  Prints a single float (mean XY distance) to stdout as the last line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a trained RL-Games policy: mean XY displacement")
parser.add_argument("--task", type=str, required=True, help="Task name (e.g., Isaac-Ant-v0, Isaac-Humanoid-v0)")
parser.add_argument("--checkpoint", type=str, required=True, help="Absolute path to RL-Games .pth checkpoint")
parser.add_argument("--eval_steps", type=int, default=2000, help="Number of evaluation steps")
parser.add_argument("--eval_num_envs", type=int, default=16, help="Number of parallel envs during eval")
parser.add_argument("--seed", type=int, default=0, help="Evaluation seed")

# Append standard AppLauncher CLI (e.g., --headless, --enable_cameras, --device)
AppLauncher.add_app_launcher_args(parser)

# Parse args; keep unknowns for Hydra
args_cli, hydra_args = parser.parse_known_args()

# Ensure headless by default if user passed --headless; we don't need cameras for eval
if getattr(args_cli, "headless", False):
    args_cli.enable_cameras = False

# Clear sys.argv for Hydra and launch app
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import math  # noqa: E402
import os  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    # Resolve checkpoint
    resume_path = retrieve_file_path(args_cli.checkpoint)
    resume_path = os.path.abspath(resume_path)

    # Apply eval overrides
    env_cfg.scene.num_envs = args_cli.eval_num_envs
    agent_cfg["params"]["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed

    # Create env (no video recording in eval)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    # Convert multi-agent to single-agent if needed
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # RL-Games vec wrapper + registry (mirrors play.py)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # Build agent and load checkpoint
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path

    # RL-Games expects the number of actors to be specified
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    # Deterministic policy for evaluation
    if hasattr(agent, "is_deterministic"):
        agent.is_deterministic = True

    # Reset env and capture initial XY positions in env frame
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs.get("obs", obs)

    # Enable batched obs and init RNN if used
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    # Position in env frame (remove per-env origin), keep only XY
    start_xy = (robot.data.root_pos_w - unwrapped.scene.env_origins)[:, :2].clone()

    # Rollout loop
    steps = int(args_cli.eval_steps)
    for _ in range(steps):
        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=True)
            obs, _, dones, _ = env.step(actions)
            if isinstance(obs, dict):
                obs = obs.get("obs", obs)
            # Reset RNN states for terminated envs
            if len(dones) > 0 and agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, dones, :] = 0.0

    end_xy = (robot.data.root_pos_w - unwrapped.scene.env_origins)[:, :2]

    # Mean L2 displacement over envs
    mean_dist = torch.linalg.vector_norm(end_xy - start_xy, dim=1).mean().item()

    # Print ONLY the float as the last line (logs may appear before from the simulator)
    print(f"{mean_dist}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure simulator closes even on exceptions
        try:
            simulation_app.close()
        except Exception:
            pass


