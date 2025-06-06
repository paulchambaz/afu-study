import torch
from omegaconf import OmegaConf
import gymnasium as gym
import torch.nn as nn
import numpy as np
from bbrl.agents import Agent  # type: ignore
from bbrl_utils.nn import build_mlp  # type: ignore
from bbrl.workspace import Workspace  # type: ignore
from .memory import ReplayBuffer
import math


class QNetwork(Agent):
    """Neural network that estimates Q-value for state action pairs."""

    def __init__(
        self, state_dim: int, hidden_dims: list[int], action_dim: int, prefix: str
    ) -> None:
        super().__init__()
        self.prefix = prefix

        # build mlp that takes concatenated state and action as input and outputs a single Q-value
        self.model = build_mlp(
            [state_dim + action_dim] + hidden_dims + [1], activation=nn.ReLU()
        )

    def forward(self, t: int) -> None:
        """Compute Q-value for a given state-action pair."""
        # get state and action from workspace
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        state_action = torch.cat([obs, action], dim=1)

        # compute Q-value
        q_value = self.model(state_action).squeeze(-1)
        self.set((f"{self.prefix}/q_value", t), q_value)


class VNetwork(Agent):
    """Neural network that estimates V-value for states."""

    def __init__(self, state_dim: int, hidden_dims: list[int], prefix: str) -> None:
        super().__init__()
        self.prefix = prefix

        # build mlp that takes state as input and outputs a single V-value
        self.model = build_mlp([state_dim] + hidden_dims + [1], activation=nn.ReLU())

    def forward(self, t: int) -> None:
        """Compute V-value for a given state."""
        # get state from workspace
        obs = self.get(("env/env_obs", t))

        # compute V-value
        v_value = self.model(obs).squeeze(-1)
        self.set((f"{self.prefix}/v_value", t), v_value)


class PolicyNetwork(Agent):
    """Neural network that outputs a Gaussian distribution over actions."""

    def __init__(self, state_dim: int, hidden_dims: list[int], action_dim: int) -> None:
        """Initialize gaussian policy network with given dimensions."""
        super().__init__()

        self.model = build_mlp(
            [state_dim] + hidden_dims + [action_dim], activation=nn.ReLU()
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, t: int) -> None:
        """Compute mean and log_std of action distribution for a given state."""
        # get state from workspace
        obs = self.get(("env/env_obs", t))

        # compute mean from the network
        mean = self.model(obs)
        log_std = self.log_std.expand_as(mean)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        self.set(("mean", t), mean)
        self.set(("log_std", t), log_std)

    def sample_action(self, workspace: Workspace, t: int) -> None:
        """Sample action from the Gaussian distribution using reparameterization trick."""
        self.workspace = workspace

        mean = self.get(("mean", t))
        log_std = self.get(("log_std", t))
        std = log_std.exp()

        # reparameterization trick
        normal = torch.randn_like(mean)
        sample = mean + std * normal

        # apply tanh squashing to bound actions
        action = torch.tanh(sample)

        self.set(("sample", t), sample)
        self.set(("action", t), action)

    def get_log_prob(self, workspace: Workspace, t: int) -> None:
        """Compute log probability of an action under the Gaussian distribution."""
        self.workspace = workspace

        mean = self.get(("mean", t))
        log_std = self.get(("log_std", t))
        sample = self.get(("sample", t))
        action = self.get(("action", t))

        normal_log_prob = (
            -0.5 * ((sample - mean) / log_std.exp()).pow(2)
            - log_std
            - 0.5 * math.log(2 * math.pi)
        ).sum(dim=-1)

        log_prob = normal_log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        self.set(("log_prob", t), log_prob)


class SACOpti:
    def __init__(self, hyperparameters: OmegaConf, optimal_model: dict):
        self.params = hyperparameters
        self.train_env = gym.make(self.params.env_name)

        self.state_dim = self.train_env.observation_space.shape[0]
        self.action_dim = self.train_env.action_space.shape[0]

        hidden_dims = [self.params.hidden_size, self.params.hidden_size]

        # self.optimal_q_network1 = optimal_model["q_network1_state"]
        # self.optimal_q_network2 = optimal_model["q_network2_state"]
        # self.optimal_v_network = optimal_model["v_network_state"]

        # Q network
        self.q_network1 = QNetwork(
            state_dim=self.state_dim,
            hidden_dims=hidden_dims,
            action_dim=self.action_dim,
            prefix="q1",
        )

        self.q_network2 = QNetwork(
            state_dim=self.state_dim,
            hidden_dims=hidden_dims,
            action_dim=self.action_dim,
            prefix="q2",
        )

        # Value networks
        self.v_network = VNetwork(
            state_dim=self.state_dim, hidden_dims=hidden_dims, prefix="v"
        )

        # Value target networks
        self.v_target_network = VNetwork(
            state_dim=self.state_dim, hidden_dims=hidden_dims, prefix="v_target"
        )

        for target_param, param in zip(
            self.v_target_network.parameters(), self.v_network.parameters()
        ):
            target_param.data.copy_(param.data)

        # Policy network
        self.policy_network = PolicyNetwork(
            state_dim=self.state_dim,
            hidden_dims=hidden_dims,
            action_dim=self.action_dim,
        )

        # Temperature
        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))

        # Setup optimizers with learning rates
        self.q1_optimizer = torch.optim.Adam(
            self.q_network1.parameters(), lr=self.params.q_lr
        )
        self.q2_optimizer = torch.optim.Adam(
            self.q_network2.parameters(), lr=self.params.q_lr
        )

        self.v_optimizer = torch.optim.Adam(
            self.v_network.parameters(), lr=self.params.v_lr
        )

        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=self.params.policy_lr
        )

        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.params.alpha_lr
        )

        self.replay_buffer = ReplayBuffer(self.params.replay_size)
        self.batch_size = self.params.batch_size
        self.target_entropy = -self.action_dim
        self.tau = self.params.tau
        self.gamma = self.params.gamma
        self.total_steps = 0

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

    def update(self):
        """Update all networks based on sampled experience from the replay buffer."""
        # Check if we have enough data for a batch
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size,
            continuous=True,
        )

        # Update Q-network 1
        q1_loss = self._compute_q_loss(
            states, actions, rewards, next_states, dones, network_idx=1
        )
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        # Update Q-network 2
        q2_loss = self._compute_q_loss(
            states, actions, rewards, next_states, dones, network_idx=2
        )
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update V-network
        va_loss = self._compute_v_loss(states, actions, rewards, next_states, dones)
        self.v_optimizer.zero_grad()
        va_loss.backward()
        self.v_optimizer.step()

        # Soft update of targets network
        self._soft_update(self.v_network, self.v_target_network)

        # Update policy network
        policy_loss = self._compute_policy_loss(states)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update alpha parameter
        alpha_loss = self._compute_alpha_loss(states)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

    def _compute_v_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the V-network loss based on the formula.

        L_V (psi) = mean [ (V_psi (s) - (min Q_theta_i (s, a_s) - alpha * log(policy_phi (a_s | s))))^2 ]
        """
        # Compute current V-values : V_psi (s)
        v_workspace = Workspace()
        v_workspace.set("env/env_obs", 0, states)
        self.v_network(v_workspace, t=0)
        v_values = v_workspace.get("v/v_value", 0)

        # Sample actions from the policy for the given states : a_s, policy_phi.alt (a_s | s)
        policy_workspace = Workspace()
        policy_workspace.set("env/env_obs", 0, states)
        self.policy_network(policy_workspace, t=0)
        self.policy_network.sample_action(policy_workspace, t=0)
        self.policy_network.get_log_prob(policy_workspace, t=0)
        actions = policy_workspace.get("action", 0)
        log_probs = policy_workspace.get("log_prob", 0)

        # Compute Q-values 1 from the sampled actions : Q_(theta_1) (s, a_s)
        q1_workspace = Workspace()
        q1_workspace.set("env/env_obs", 0, states)
        q1_workspace.set("action", 0, actions)
        self.q_network1(q1_workspace, t=0)
        q1_values = q1_workspace.get("q1/q_value", 0)

        # Compute Q-values 2 from the sampled actions : Q_(theta_1) (s, a_s)
        q2_workspace = Workspace()
        q2_workspace.set("env/env_obs", 0, states)
        q2_workspace.set("action", 0, actions)
        self.q_network2(q2_workspace, t=0)
        q2_values = q2_workspace.get("q2/q_value", 0)

        # Use minimum of the Q-values to reduce overestimation bias : min Q_(theta_i) (s, a_s)
        min_q_values = torch.min(q1_values, q2_values)

        # Compute the target : min Q_(theta_i) (s, a_s) - alpha * log(policy_phi.alt (a_s | s))
        alpha = self.log_alpha.exp().detach()
        v_targets = min_q_values - alpha * log_probs

        # Compute the mean squared error loss : L_V (psi)
        v_loss = 0.5 * torch.mean((v_values - v_targets.detach()) ** 2)

        return v_loss

    def _compute_q_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        network_idx: int,
    ) -> torch.Tensor:
        """
        Compute the Q-network loss based on the Bellman equation.

        L_Q (theta_i) = mean [ (Q_(theta_i) (s, a) - r - gamma V_(psi^target) (s'))^2 ]
        """
        if network_idx == 1:
            q_network = self.q_network1
            prefix = "q1"
        else:
            q_network = self.q_network2
            prefix = "q2"

        # Compute Q-values : Q_(theta_i) (s, a)
        q_workspace = Workspace()
        q_workspace.set("env/env_obs", 0, states)
        q_workspace.set("action", 0, actions)
        q_network(q_workspace, t=0)
        q_values = q_workspace.get(f"{prefix}/q_value", 0)

        # Compute target V-values : V_(psi^target) (s')
        target_workspace = Workspace()
        target_workspace.set("env/env_obs", 0, next_states)
        self.v_target_network(target_workspace, t=0)
        v_targets = target_workspace.get("v_target/v_value", 0)

        # Compute the targets : r + gamma * V_(psi^target) (s')
        q_targets = rewards + (1.0 - dones) * self.gamma * v_targets

        # Compute the mean squared error loss : L_Q (theta)
        q_loss = 0.5 * torch.mean((q_values - q_targets.detach()) ** 2)

        return q_loss

    def _compute_policy_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute the policy loss based on the formula.

        L_policy (phi.alt) = mean [ alpha * log(policy_(phi.alt) (a_s | s)) - Q_(theta_i) (s, a_s) ]
        """
        # Create a workspace to compute policy distribution : a_s, policy_phi.alt (a_s | s)
        policy_workspace = Workspace()
        policy_workspace.set("env/env_obs", 0, states)
        self.policy_network(policy_workspace, t=0)
        self.policy_network.sample_action(policy_workspace, t=0)
        self.policy_network.get_log_prob(policy_workspace, t=0)
        actions = policy_workspace.get("action", 0)
        log_probs = policy_workspace.get("log_prob", 0)

        # Compute Q-values 1 for these sampled actions
        q1_workspace = Workspace()
        q1_workspace.set("env/env_obs", 0, states)
        q1_workspace.set("action", 0, actions)
        self.q_network1(q1_workspace, t=0)
        q1_values = q1_workspace.get("q1/q_value", 0)

        # Compute Q-values 2 for these sampled actions
        q2_workspace = Workspace()
        q2_workspace.set("env/env_obs", 0, states)
        q2_workspace.set("action", 0, actions)
        self.q_network2(q2_workspace, t=0)
        q2_values = q2_workspace.get("q2/q_value", 0)

        # Use minimum of the Q-values to reduce overestimation bias : min Q_(theta_i) (s, a_s)
        min_q_values = torch.min(q1_values, q2_values)

        # Compute the policy loss : L_policy (phi.alt)
        alpha = self.log_alpha.exp()
        policy_loss = (alpha * log_probs - min_q_values).mean()

        return policy_loss

    def _compute_alpha_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute the temperature parameter loss.

        L_temp (alpha) = [ -log (policy_theta (a_s | s)) - alpha target_entropy ]
        """
        # Create a workspace to compute policy distribution : log policy_theta (a_s | s)
        policy_workspace = Workspace()
        policy_workspace.set("env/env_obs", 0, states)
        self.policy_network(policy_workspace, t=0)

        # Sample actions from the policy
        self.policy_network.sample_action(policy_workspace, t=0)
        self.policy_network.get_log_prob(policy_workspace, t=0)
        log_probs = policy_workspace.get("log_prob", 0)

        # Compute the alpha loss : L_temp (alpha)
        alpha = self.log_alpha.exp()
        alpha_loss = (-alpha * (log_probs + self.target_entropy)).mean()

        return alpha_loss

    def _soft_update(
        self, source_network: nn.Module, target_network: nn.Module
    ) -> None:
        """Perform soft update of target network parameters."""
        for target_param, source_param in zip(
            target_network.parameters(), source_network.parameters()
        ):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * source_param.data
            )

    def get_weights(self) -> dict:
        return {
            # Network states
            "q_network1_state": self.q_network1.state_dict(),
            "q_network2_state": self.q_network2.state_dict(),
            "v_network_state": self.v_network.state_dict(),
            "v_target_network_state": self.v_target_network.state_dict(),
            "policy_network_state": self.policy_network.state_dict(),
            # Optimizer states
            "q1_optimizer_state": self.q1_optimizer.state_dict(),
            "q2_optimizer_state": self.q2_optimizer.state_dict(),
            "v_optimizer_state": self.v_optimizer.state_dict(),
            "policy_optimizer_state": self.policy_optimizer.state_dict(),
            "alpha_optimizer_state": self.alpha_optimizer.state_dict(),
            # Temperature parameter
            "log_alpha": self.log_alpha.detach().cpu(),
            # Other parameters
            "params": self.params,
        }

    def save(self, path: str) -> None:
        save_dict = self.get_weights()
        torch.save(save_dict, path)

    def load(self, path: str) -> None:
        save_dict = torch.load(path)

        # Restore network states
        self.q_network1.load_state_dict(save_dict["q_network1_state"])
        self.q_network2.load_state_dict(save_dict["q_network2_state"])
        self.v_network.load_state_dict(save_dict["v_network_state"])
        self.v_target_network.load_state_dict(save_dict["v_target_network_state"])
        self.policy_network.load_state_dict(save_dict["policy_network_state"])

        # Restore optimizer states
        self.q1_optimizer.load_state_dict(save_dict["q1_optimizer_state"])
        self.q2_optimizer.load_state_dict(save_dict["q2_optimizer_state"])
        self.v_optimizer.load_state_dict(save_dict["v_optimizer_state"])
        self.policy_optimizer.load_state_dict(save_dict["policy_optimizer_state"])
        self.alpha_optimizer.load_state_dict(save_dict["alpha_optimizer_state"])

        # Restore temperature parameter
        with torch.no_grad():
            self.log_alpha.copy_(save_dict["log_alpha"])

        # Restore other parameters
        self.params = save_dict["params"]

    @classmethod
    def loadagent(cls, path: str) -> "SACOpti":
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
                "v_lr": 3e-4,
                "policy_lr": 3e-4,
                "alpha_lr": 3e-4,
                "replay_size": 200_000,
                "batch_size": 256,
                "tau": 0.01,
                "gamma": 0.99,
            }
        )

    @classmethod
    def _get_hp_space(cls):
        return {
            "hidden_size": ("int", 32, 512, True),
            "q_lr": ("float", 1e-5, 1e-2, True),
            "v_lr": ("float", 1e-5, 1e-2, True),
            "policy_lr": ("float", 1e-5, 1e-2, True),
            "alpha_lr": ("float", 1e-5, 1e-2, True),
            "replay_size": ("int", 10_000, 1_000_000, True),
            "batch_size": ("int", 32, 1024, True),
            "tau": ("float", 1e-4, 1e-1, True),
            "gamma": ("float", 0.9, 0.999, False),
        }
