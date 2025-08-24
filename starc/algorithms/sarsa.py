import random
import os
from collections import namedtuple, deque
from itertools import count
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from starc.core.half_cheetah_env import HalfCheetahEnv
from starc.utils.simple_utils import sample_space, timed

# Device selection with override via env var (supports "cpu", "cuda", or "cuda:<idx>")
_env_override = os.getenv("STARC_SARSA_DEVICE", None)
if _env_override:
    try:
        # Allow explicit GPU index: e.g., "cuda:1"
        if _env_override.startswith("cuda") and not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device(_env_override)
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    # default: auto
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class SARSA(nn.Module):

    # !!!!!!! changed from n_actions to 1

    def __init__(self, n_observations):
        super(SARSA, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        # self.layer2 = nn.Linear(128, 256)
        # self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        # x = F.relu(self.layer3(x))
        return self.layer4(x)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
_bs_override = os.getenv("STARC_SARSA_BATCH_SIZE")
try:
    BATCH_SIZE = int(_bs_override) if _bs_override else 128
except Exception:
    BATCH_SIZE = 128
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
_lr_override = os.getenv("STARC_SARSA_LR")
try:
    LR = float(_lr_override) if _lr_override else 1e-4
except Exception:
    LR = 1e-4

def select_action():
    # Sample action as float32 numpy array for continuous control
    return np.array(sample_space(HalfCheetahEnv.act_space), dtype=np.float32)

def optimize_model(value_net, optimizer, memory, env):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    # action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_values = value_net(state_batch) #.gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, dtype=torch.float32, device=device)
    with torch.no_grad():
        
        next_state_values[non_final_mask] = value_net(non_final_next_states).squeeze()
    # Compute the expected Q values
    discount_tensor = torch.tensor(env.discount, dtype=torch.float32, device=device)
    expected_state_values = (next_state_values * discount_tensor) + reward_batch.squeeze()

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_values.squeeze(), expected_state_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(value_net.parameters(), 100)
    optimizer.step()

@timed
def train_sarsa_model(env, class_name: str, n_episodes: int = 10000):
    # Progress logging control
    progress_enabled = os.getenv("STARC_SARSA_PROGRESS", "0") == "1"
    if progress_enabled:
        print(f"[SARSA] Starting StateVals training for {class_name} | device={device} | episodes={n_episodes} | batch_size={BATCH_SIZE} | lr={LR}")
        log_every = max(1, n_episodes // 20)
    # Determine input dimension directly from the environment to stay robust
    # to changes such as including or excluding the root x-position.
    obs_dim = env.observation_space.shape[0]
    value_net = SARSA(obs_dim).to(device)

    # Try multiple cache locations for pre-trained models
    # Namespace by environment if provided to avoid cross-env reuse
    namespace = os.getenv("STARC_SARSA_NAMESPACE")
    if namespace:
        rel_path = f"sarsa_models/{namespace}/{class_name}/{n_episodes}.pt"
    else:
        rel_path = f"sarsa_models/{class_name}/{n_episodes}.pt"
    candidate_paths = [
        rel_path,
        os.path.join("starc_v2", "scripts", rel_path),
        os.path.join("starc", "scripts", rel_path),
    ]
    # Optional purge of existing checkpoints
    if os.getenv("STARC_SARSA_PURGE_CACHE", "0") == "1":
        for file_path in candidate_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass
    # Load from cache unless forced to retrain
    if os.getenv("STARC_SARSA_FORCE_RETRAIN", "0") != "1":
        for file_path in candidate_paths:
            if os.path.exists(file_path):
                value_net.load_state_dict(torch.load(file_path, map_location=device))
                return value_net
    
    optimizer = optim.AdamW(value_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = 0

    for i_episode in range(n_episodes):
        # Initialize the environment and get its state
        state = env.reset()
        if isinstance(state, tuple):  # Unpack if needed (Gymnasium API)
            state = state[0]
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action()
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            reward = torch.tensor([reward], dtype=torch.float32, device=device)

            if done:
                next_state = None
            else:
                next_state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(value_net, optimizer, memory, env)

            if done or t > 100:
                break

        if progress_enabled and (i_episode % log_every == 0 or i_episode == n_episodes - 1):
            print(f"[SARSA] {class_name}: episode {i_episode+1}/{n_episodes} | memory={len(memory)}")

    # save the model (write to primary relative dir and starc_v2/scripts mirror)
    # Mirror save to namespaced directories
    namespace = os.getenv("STARC_SARSA_NAMESPACE")
    if namespace:
        primary_dir = os.path.join('sarsa_models', namespace, class_name)
        mirror_dir = os.path.join('starc_v2', 'scripts', 'sarsa_models', namespace, class_name)
    else:
        primary_dir = os.path.join('sarsa_models', class_name)
        mirror_dir = os.path.join('starc_v2', 'scripts', 'sarsa_models', class_name)
    os.makedirs(primary_dir, exist_ok=True)
    os.makedirs(mirror_dir, exist_ok=True)
    torch.save(value_net.state_dict(), os.path.join(primary_dir, f"{n_episodes}.pt"))
    try:
        torch.save(value_net.state_dict(), os.path.join(mirror_dir, f"{n_episodes}.pt"))
    except Exception:
        pass

    return value_net
