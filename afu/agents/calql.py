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


# Q-network to estimate Q-values for state-action pairs
class QNetwork(Agent):
    """Neural network that estimates Q-values for state-action pairs."""
    def __init__(self, state_dim: int, hidden_dims: list[int], action_dim: int, prefix: str):
        super().__init__()
        self.prefix = prefix
        # Multi-layer perceptron for Q-value approximation
        self.model = build_mlp([state_dim + action_dim] + hidden_dims + [1], activation=nn.ReLU())

    def forward(self, t: int):
        # Extract state and action from workspace
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        state_action = torch.cat([obs, action], dim=1)
        # Compute Q-value
        q_value = self.model(state_action).squeeze(-1)
        self.set((f"{self.prefix}/q_value", t), q_value)

# Policy network that outputs a Gaussian distribution over actions
class PolicyNetwork(Agent):
    """Gaussian policy network for action selection."""
    def __init__(self, state_dim: int, hidden_dims: list[int], action_dim: int):
        super().__init__()
        # Multi-layer perceptron for policy network
        self.model = build_mlp([state_dim] + hidden_dims + [action_dim], activation=nn.ReLU())
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable standard deviation
        self.log_std_min, self.log_std_max = -20, 2

    def forward(self, t: int):
        # Compute mean and log standard deviation for the action distribution
        obs = self.get(("env/env_obs", t))
        mean = self.model(obs)
        log_std = torch.clamp(self.log_std.expand_as(mean), self.log_std_min, self.log_std_max)
        self.set(("mean", t), mean)
        self.set(("log_std", t), log_std)

    def sample_action(self, workspace: Workspace, t: int):
        # Sample action using reparameterization trick
        self.workspace = workspace
        mean, log_std = self.get(("mean", t)), self.get(("log_std", t))
        std = log_std.exp()
        normal = torch.randn_like(mean)
        sample = mean + std * normal
        action = torch.tanh(sample)  # Squash action to bounded range
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
        
        # Initialize Q-networks and policy network
        self.q_network1 = QNetwork(self.state_dim, hidden_dims, self.action_dim, "q1")
        self.q_network2 = QNetwork(self.state_dim, hidden_dims, self.action_dim, "q2")
        self.policy_network = PolicyNetwork(self.state_dim, hidden_dims, self.action_dim)
        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        
        # Optimizers for Q-networks, policy network, and entropy coefficient
        self.q1_optimizer = optim.Adam(self.q_network1.parameters(), lr=self.params.q_lr)
        self.q2_optimizer = optim.Adam(self.q_network2.parameters(), lr=self.params.q_lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.params.policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.params.alpha_lr)
        
        self.replay_buffer = ReplayBuffer(self.params.replay_size)
        self.batch_size = self.params.batch_size
        self.gamma = self.params.gamma
        self.cql_alpha = self.params.cql_alpha
        self.total_steps = 0

    def select_action(self, state: np.ndarray, evaluation: bool = False) -> np.ndarray:
        """Selects an action using the policy network."""
        workspace = Workspace()
        state_tensor = torch.FloatTensor(state[None, ...])
        workspace.set("env/env_obs", 0, state_tensor)
        
        self.policy_network(workspace, t=0)
        
        if evaluation:
            action = workspace.get("mean", 0)  # Use mean action during evaluation
        else:
            self.policy_network.sample_action(workspace, t=0)
            action = workspace.get("action", 0)  # Sampled action during training
        
        return action.detach().numpy()[0]

    def update(self):
        """Performs a full update step for Q-network, policy network, and entropy coefficient."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a batch from the replay buffer, including Monte Carlo returns
        states, actions, rewards, next_states, dones, mc_returns = self.replay_buffer.sample(
            self.batch_size,
            continuous=True,
        )
        
        # Update Q-network
        q_loss = self._compute_q_loss(states, actions, rewards, next_states, dones, mc_returns)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Update policy network
        policy_loss = self._compute_policy_loss(states)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update entropy coefficient alpha
        alpha_loss = self._compute_alpha_loss(states)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
    
    def _compute_q_loss(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor, mc_returns: torch.Tensor) -> torch.Tensor:
        """Computes the Q-network loss using the conservative Q-learning formulation with Monte Carlo returns."""
        # Compute current Q-values for (s, a)
        q_values = self.q_network(states, actions)
        
        # Compute target Q-values using Bellman equation
        with torch.no_grad():
            next_actions, log_pis = self.policy_network.sample_action(next_states)
            target_q_values = self.q_network(next_states, next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * target_q_values
        
        # Ensure Q-values are at least as large as Monte Carlo returns
        q_values = torch.max(q_values, mc_returns)
        
        # Compute conservative Q-loss using log-sum-exp trick
        num_samples = 4
        random_actions = torch.rand((num_samples, self.batch_size, self.action_dim)) * 2 - 1
        q_rand = self.q_network(states.repeat(num_samples, 1, 1), random_actions)
        q_rand = q_rand.view(num_samples, self.batch_size, -1).mean(dim=0)
        
        td_loss = F.mse_loss(q_values, target_q)
        cql_loss = ((torch.logsumexp(q_values, dim=0) - q_values).mean() * self.cql_alpha) + q_rand.mean()
        
        return td_loss + cql_loss
    
    def _compute_policy_loss(self, states: torch.Tensor) -> torch.Tensor:
        """Computes the policy loss using entropy regularization and Q-value feedback."""
        actions, log_pis = self.policy_network.sample_action(states)
        q_values = self.q_network(states, actions)
        policy_loss = (self.log_alpha.exp() * log_pis - q_values).mean()
        return policy_loss
    
    def _compute_alpha_loss(self, states: torch.Tensor) -> torch.Tensor:
        """Computes the temperature coefficient loss for entropy regularization."""
        _, log_pis = self.policy_network.sample_action(states)
        alpha_loss = (-self.log_alpha.exp() * (log_pis + self.params.target_entropy)).mean()
        return alpha_loss

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