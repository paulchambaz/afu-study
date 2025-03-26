import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf
from bbrl.agents import Agent
from bbrl.workspace import Workspace
from bbrl_utils.nn import build_mlp
from .memory import ReplayBuffer
import torch.nn.functional as F

class QNetwork(Agent):
    """Neural network that estimates Q-values for state-action pairs."""
    def __init__(self, state_dim: int, hidden_dims: list[int], action_dim: int, prefix: str):
        super().__init__()
        self.prefix = prefix
        self.model = build_mlp([state_dim + action_dim] + hidden_dims + [1], activation=nn.ReLU())

    def forward(self, t: int):
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        state_action = torch.cat([obs, action], dim=1)
        q_value = self.model(state_action).squeeze(-1)
        self.set((f"{self.prefix}/q_value", t), q_value)

class PolicyNetwork(Agent):
    """Gaussian policy network for action selection."""
    def __init__(self, state_dim: int, hidden_dims: list[int], action_dim: int):
        super().__init__()
        self.model = build_mlp([state_dim] + hidden_dims + [action_dim], activation=nn.ReLU())
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min, self.log_std_max = -20, 2

    def forward(self, t: int):
        obs = self.get(("env/env_obs", t))
        mean = self.model(obs)
        log_std = torch.clamp(self.log_std.expand_as(mean), self.log_std_min, self.log_std_max)
        self.set(("mean", t), mean)
        self.set(("log_std", t), log_std)

    def sample_action(self, workspace: Workspace, t: int):
        mean, log_std = self.get(("mean", t)), self.get(("log_std", t))
        std = log_std.exp()
        normal = torch.randn_like(mean)
        sample = mean + std * normal
        action = torch.tanh(sample)
        self.set(("sample", t), sample)
        self.set(("action", t), action)

class CalQL:
    """Calibrated Q-learning agent."""
    def __init__(self, hyperparameters: OmegaConf):
        self.params = hyperparameters
        self.train_env = gym.make(self.params.env_name)
        self.state_dim = self.train_env.observation_space.shape[0]
        self.action_dim = self.train_env.action_space.shape[0]
        hidden_dims = [self.params.hidden_size] * 2
        
        self.q_network1 = QNetwork(self.state_dim, hidden_dims, self.action_dim, "q1")
        self.q_network2 = QNetwork(self.state_dim, hidden_dims, self.action_dim, "q2")
        self.policy_network = PolicyNetwork(self.state_dim, hidden_dims, self.action_dim)
        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        
        self.q1_optimizer = optim.Adam(self.q_network1.parameters(), lr=self.params.q_lr)
        self.q2_optimizer = optim.Adam(self.q_network2.parameters(), lr=self.params.q_lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.params.policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.params.alpha_lr)
        
        self.replay_buffer = ReplayBuffer(self.params.replay_size)
        self.batch_size = self.params.batch_size
        self.gamma = self.params.gamma
        self.cql_alpha = self.params.cql_alpha
        self.total_steps = 0

    def _compute_q_loss(self, workspace: Workspace):
        obs = workspace.get("env/env_obs", 0)
        actions = workspace.get("action", 0)
        rewards = workspace.get("reward", 0)
        dones = workspace.get("done", 0)
        mc_returns = workspace.get("mc_return", 0)
        
        next_workspace = Workspace()
        next_workspace.set("env/env_obs", 0, workspace.get("env/next_obs", 0))
        self.policy_network(next_workspace, t=0)
        self.policy_network.sample_action(next_workspace, t=0)
        next_actions = next_workspace.get("action", 0)
        
        self.q_network1(next_workspace, t=0)
        self.q_network2(next_workspace, t=0)
        next_q1 = next_workspace.get("q1/q_value", 0)
        next_q2 = next_workspace.get("q2/q_value", 0)
        target_qval = rewards + (1.0 - dones) * self.gamma * torch.min(next_q1, next_q2)
        
        self.q_network1(workspace, t=0)
        self.q_network2(workspace, t=0)
        q1_values = workspace.get("q1/q_value", 0)
        q2_values = workspace.get("q2/q_value", 0)
        
        td_loss = F.mse_loss(q1_values, target_qval) + F.mse_loss(q2_values, target_qval)
        critic_loss = td_loss + ((torch.logsumexp(q1_values, dim=0) - q1_values).mean() * self.cql_alpha)
        return critic_loss

    def _compute_policy_loss(self, workspace: Workspace):
        self.policy_network(workspace, t=0)
        self.policy_network.sample_action(workspace, t=0)
        log_pis = workspace.get("log_prob", 0)
        q_values = workspace.get("q1/q_value", 0)
        policy_loss = (log_pis * self.log_alpha.exp() - q_values).mean()
        return policy_loss

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        workspace = Workspace()
        batch = self.replay_buffer.sample(self.batch_size, continuous=True)
        workspace.set("env/env_obs", 0, batch["observations"])
        workspace.set("action", 0, batch["actions"])
        workspace.set("reward", 0, batch["rewards"])
        workspace.set("done", 0, batch["dones"])
        workspace.set("mc_return", 0, batch["mc_return"])
        workspace.set("env/next_obs", 0, batch["next_observations"])
        
        q_loss = self._compute_q_loss(workspace)
        self.q1_optimizer.zero_grad()
        q_loss.backward()
        self.q1_optimizer.step()
        
        policy_loss = self._compute_policy_loss(workspace)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def select_action(self, state: np.ndarray, evaluation: bool = False) -> np.ndarray:
        workspace = Workspace()
        state_tensor = torch.FloatTensor(state[None, ...])
        workspace.set("env/env_obs", 0, state_tensor)
        self.policy_network(workspace, t=0)
        if evaluation:
            action = workspace.get("mean", 0)
        else:
            self.policy_network.sample_action(workspace, t=0)
            action = workspace.get("action", 0)
        return action.detach().numpy()[0]
    
    def get_weights(self) -> dict:
        return {
            # Network states
            "q_network1_state": self.q_network1.state_dict(),
            "q_network2_state": self.q_network2.state_dict(),
            "policy_network_state": self.policy_network.state_dict(),
            # Optimizer states
            "q1_optimizer_state": self.q1_optimizer.state_dict(),
            "q2_optimizer_state": self.q2_optimizer.state_dict(),
            "policy_optimizer_state": self.policy_optimizer.state_dict(),
            "alpha_optimizer_state": self.alpha_optimizer.state_dict(),
            # Temperature parameter
            "log_alpha": self.log_alpha.detach().cpu(),
            # Other parameters
            "params": self.params,
            "total_steps": self.total_steps,
        }

    def save(self, path: str) -> None:
        save_dict = self.get_weights()
        torch.save(save_dict, path)

    def load(self, path: str) -> None:
        save_dict = torch.load(path)

        # Restore network states
        self.q_network1.load_state_dict(save_dict["q_network1_state"])
        self.q_network2.load_state_dict(save_dict["q_network2_state"])
        self.policy_network.load_state_dict(save_dict["policy_network_state"])

        # Restore optimizer states
        self.q1_optimizer.load_state_dict(save_dict["q1_optimizer_state"])
        self.q2_optimizer.load_state_dict(save_dict["q2_optimizer_state"])
        self.policy_optimizer.load_state_dict(save_dict["policy_optimizer_state"])
        self.alpha_optimizer.load_state_dict(save_dict["alpha_optimizer_state"])

        # Restore temperature parameter
        with torch.no_grad():
            self.log_alpha.copy_(save_dict["log_alpha"])

        # Restore other parameters
        self.params = save_dict["params"]
        self.total_steps = save_dict["total_steps"]

    @classmethod
    def loadagent(cls, path: str) -> "CalQL":
        save_dict = torch.load(path)
        agent = cls(save_dict["params"])
        agent.load(path)
        return agent

    @classmethod
    def _get_params_defaults(cls) -> OmegaConf:
        return OmegaConf.create(
            {
                "hidden_size": 256,
                "q_lr": 3e-4,
                "policy_lr": 3e-4,
                "alpha_lr": 3e-4,
                "replay_size": 100_000,
                "batch_size": 256,
                "gamma": 0.99,
                "cql_alpha": 1.0,
            }
        )

    @classmethod
    def _get_hp_space(cls):
        return {
            "hidden_size": ("int", 32, 512, True),
            "q_lr": ("float", 1e-5, 1e-2, True),
            "policy_lr": ("float", 1e-5, 1e-2, True),
            "alpha_lr": ("float", 1e-5, 1e-2, True),
            "replay_size": ("int", 10_000, 1_000_000, True),
            "batch_size": ("int", 32, 1024, True),
            "gamma": ("float", 0.9, 0.999, False),
            "cql_alpha": ("float", 0.1, 10.0, True),
        }