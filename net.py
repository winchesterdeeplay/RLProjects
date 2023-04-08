import numpy as np
import torch
from gymnasium import Env
from gymnasium.core import ObsType


class Net(torch.nn.Module):
    """Simple MLP network for Q-learning."""

    def __init__(self, env_state: Env, net_linear_hidden_size: int = 64) -> None:
        super(Net, self).__init__()
        in_features = int(np.prod(env_state.observation_space.shape))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, net_linear_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(net_linear_hidden_size, env_state.action_space.n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def action(self, observation: ObsType) -> int:
        obs_t = torch.tensor(observation, dtype=torch.float32)
        q_values = self.forward(obs_t.unsqueeze(0))

        max_q_index = q_values.argmax(dim=1).item()
        return max_q_index
