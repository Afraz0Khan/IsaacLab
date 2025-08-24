import torch

class StateVals:
  """
  A class for computing the state values of a PPO agent
  """
  def __init__(self,
               env,  # Remove type hint to avoid circular import
               reward_class_name: str,
               n_episodes_sarsa: int = 10000):
    # Lazy import to avoid circular dependency
    from starc.algorithms.sarsa import train_sarsa_model, device as sarsa_device
    
    self.model = train_sarsa_model(env, reward_class_name, n_episodes_sarsa)
    self.device = sarsa_device
    self.model.eval()

    for param in self.model.parameters():
      param.requires_grad = False

  def __call__(self, state):
    obs_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
    with torch.no_grad():
      return self.model(obs_tensor).item()

  def evaluate_batch(self, states, batch_size: int = 1024):
    """Evaluate V(s) for a batch of states. Returns a NumPy float32 array.

    Args:
      states: array-like [N, obs_dim]
      batch_size: micro-batch size for memory control
    """
    import numpy as np
    states = np.asarray(states, dtype=np.float32)
    values = np.empty((states.shape[0],), dtype=np.float32)
    with torch.no_grad():
      for start in range(0, states.shape[0], batch_size):
        end = min(start + batch_size, states.shape[0])
        obs_tensor = torch.from_numpy(states[start:end]).to(self.device)
        out = self.model(obs_tensor).view(-1)
        values[start:end] = out.detach().cpu().numpy().astype(np.float32)
    return values
