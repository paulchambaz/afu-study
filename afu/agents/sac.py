import torch  # type: ignore
import torch.nn as nn  # type: ignore
from bbrl.agents import Agent  # type: ignore
from bbrl.utils.nn import build_mlp  # type: ignore


class ActorAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.model = build_mlp(
            [state_dim] + list(hidden_layers) + [action_dim * 2], activation=nn.ReLU()
        )

    def forward(self, t):
        obs = self.get(("env/env_obs", t))
        out = self.model(obs)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)
        action = torch.tanh(dist.rsample())
        log_prob = dist.log_prob(action).sum(dim=-1)

        self.set(("action", t), action)
        self.set(("action_logprobs", t), log_prob)


class CriticAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.model = build_mlp(
            [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, t):
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        x = torch.cat([obs, action], dim=-1)
        q_value = self.model(x).squeeze(-1)
        self.set(("q_value", t), q_value)
