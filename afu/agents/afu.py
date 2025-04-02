import torch
import math
from omegaconf import OmegaConf
import gymnasium as gym
import torch.nn as nn
import numpy as np
from bbrl.agents import Agent  # type: ignore
from bbrl_utils.nn import build_mlp  # type: ignore
from bbrl.workspace import Workspace  # type: ignore
from .memory import ReplayBuffer


class QNetwork(Agent):
    """Neural network that estimates Q-value for state action pairs."""

    def __init__(self, state_dim: int, hidden_dims: list[int], action_dim: int) -> None:
        super().__init__()

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
        print("before q compute")
        print(f"state action (shape={state_action.shape}): {state_action}")
        q_value = self.model(state_action).squeeze(-1)
        print("after q compute")
        self.set(("q_value", t), q_value)


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
        print("before v compute")
        print(f"obs (shape={obs.shape}): {obs}")
        v_value = self.model(obs).squeeze(-1)
        print("after v compute")
        self.set((f"{self.prefix}/v_value", t), v_value)


class ANetwork(Agent):
    """Neural network that estimates advantage values for state-action pairs."""

    def __init__(
        self, state_dim: int, hidden_dims: list[int], action_dim: int, prefix: str
    ) -> None:
        super().__init__()
        self.prefix = prefix

        self.model = build_mlp(
            [state_dim + action_dim] + hidden_dims + [1], activation=nn.ReLU()
        )

    def forward(self, t: int) -> None:
        """Compute A-value for a given state-action pair."""
        # get state and action from workspace
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        state_action = torch.cat([obs, action], dim=1)

        # compute A-value
        print("before a compute")
        print(f"state action (shape={state_action.shape}): {state_action}")
        a_value = self.model(state_action).squeeze(-1)
        print("after a compute")
        self.set((f"{self.prefix}/a_value", t), a_value)


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
        print("before policy compute")
        print(f"obs (shape={obs.shape}): {obs}")
        mean = self.model(obs)
        print("after policy compute")
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


class AFU:
    def __init__(self, hyperparameters: OmegaConf):
        self.params = hyperparameters
        self.train_env = gym.make(self.params.env_name)

        self.state_dim = self.train_env.observation_space.shape[0]
        self.action_dim = self.train_env.action_space.shape[0]

        hidden_dims = [self.params.hidden_size, self.params.hidden_size]

        # Q network
        self.q_network = QNetwork(
            state_dim=self.state_dim,
            hidden_dims=hidden_dims,
            action_dim=self.action_dim,
        )

        # Value networks
        self.v_network1 = VNetwork(
            state_dim=self.state_dim, hidden_dims=hidden_dims, prefix="v1"
        )
        self.v_network2 = VNetwork(
            state_dim=self.state_dim, hidden_dims=hidden_dims, prefix="v2"
        )

        # Value target networks
        self.v_target_network1 = VNetwork(
            state_dim=self.state_dim, hidden_dims=hidden_dims, prefix="v1_target"
        )
        self.v_target_network2 = VNetwork(
            state_dim=self.state_dim, hidden_dims=hidden_dims, prefix="v2_target"
        )

        for target_param, param in zip(
            self.v_target_network1.parameters(), self.v_network1.parameters()
        ):
            target_param.data.copy_(param.data)
        for target_param, param in zip(
            self.v_target_network2.parameters(), self.v_network2.parameters()
        ):
            target_param.data.copy_(param.data)

        # Advantage networks
        self.a_network1 = ANetwork(
            state_dim=self.state_dim,
            hidden_dims=hidden_dims,
            action_dim=self.action_dim,
            prefix="a1",
        )
        self.a_network2 = ANetwork(
            state_dim=self.state_dim,
            hidden_dims=hidden_dims,
            action_dim=self.action_dim,
            prefix="a2",
        )

        # Policy network
        self.policy_network = PolicyNetwork(
            state_dim=self.state_dim,
            hidden_dims=hidden_dims,
            action_dim=self.action_dim,
        )

        # Temperature
        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))

        # Setup optimizers with learning rates
        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.params.q_lr
        )

        self.v1_optimizer = torch.optim.Adam(
            self.v_network1.parameters(), lr=self.params.v_lr
        )
        self.v2_optimizer = torch.optim.Adam(
            self.v_network2.parameters(), lr=self.params.v_lr
        )

        self.a1_optimizer = torch.optim.Adam(
            self.a_network1.parameters(), lr=self.params.v_lr
        )
        self.a2_optimizer = torch.optim.Adam(
            self.a_network2.parameters(), lr=self.params.v_lr
        )

        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=self.params.policy_lr
        )

        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.params.alpha_lr
        )

        # Extra parameters
        self.replay_buffer = ReplayBuffer(self.params.replay_size)
        self.batch_size = self.params.batch_size
        self.target_entropy = -self.action_dim
        self.gradient_reduction = self.params.gradient_reduction
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

        print("sampled from replay buffer")

        # Update Q-network
        q_loss = self._compute_q_loss(states, actions, rewards, next_states, dones)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        print("updated q-network")

        # Update V-network 1 and A-network 1
        va_loss1 = self._compute_va_loss(
            states, actions, rewards, next_states, dones, network_idx=1
        )
        self.v1_optimizer.zero_grad()
        self.a1_optimizer.zero_grad()
        va_loss1.backward()
        self.v1_optimizer.step()
        self.a1_optimizer.step()

        print("updated v and a networks 1")

        # Update V-network 1 and A-network 2
        va_loss2 = self._compute_va_loss(
            states, actions, rewards, next_states, dones, network_idx=2
        )
        self.v2_optimizer.zero_grad()
        self.a2_optimizer.zero_grad()
        va_loss2.backward()
        self.v2_optimizer.step()
        self.a2_optimizer.step()

        print("updated v and a networks 2")

        # Soft update of targets network
        self._soft_update(self.v_network1, self.v_target_network1)
        self._soft_update(self.v_network2, self.v_target_network2)

        print("updated target v networks 1 and 2")

        # Update policy network
        policy_loss = self._compute_policy_loss(states)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        print("updated policy network")

        # Update alpha parameter
        alpha_loss = self._compute_alpha_loss(states)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        print("updated alpha parameter")

    def _compute_va_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        network_idx: int,
    ) -> torch.Tensor:
        """Compute the combine value and advantage loss.

        L_(V A) (phi_i, xi_i) = mean [ Z( Upsilon_i^a (s) - r - gamma min V_(phi_i^target) (s'), A_(xi_i) (s, a)) ]

        where Z(x, y) = (x + y)^2 if x >= 0, else x^2 + y^2

        where Upsilon_i^a(s) = (1 - rho * I_i^(s,a)) V_(phi_i) (s) + rho * I_i^(s,a) * V_(phi_i)^nograd (s)

        where I_i^(s,a) = 1 if V_(phi_i) (s) + A_(xi_i) (s,a) < Q_psi (s, a) else 0
        """
        if network_idx == 1:
            v_network = self.v_network1
            a_network = self.a_network1
            v_prefix = "v1"
            a_prefix = "a1"
        else:
            v_network = self.v_network2
            a_network = self.a_network2
            v_prefix = "v2"
            a_prefix = "a2"

        # Compute the V-value 1 : V_(phi_1^target) (s')
        target_workspace1 = Workspace()
        target_workspace1.set("env/env_obs", 0, next_states)
        self.v_target_network1(target_workspace1, t=0)
        v1_targets = target_workspace1.get("v1_target/v_value", 0)

        # Compute the V-value 2 : V_(phi_2^target) (s')
        target_workspace2 = Workspace()
        target_workspace2.set("env/env_obs", 0, next_states)
        self.v_target_network2(target_workspace2, t=0)
        v2_targets = target_workspace2.get("v2_target/v_value", 0)

        # Take the minimum of the two target networks : min V_(phi_i^target) (s')
        min_v_targets = torch.min(v1_targets, v2_targets)

        # Compute the target Q-values : r + gamma min V_(phi_i^target) (s')
        q_targets = rewards + (1.0 - dones) * self.gamma * min_v_targets

        # Compute V_(phi_i) (s) and V_(phi_i^nograd (s)
        v_workspace = Workspace()
        v_workspace.set("env/env_obs", 0, states)
        v_network(v_workspace, t=0)
        v_values = v_workspace.get(f"{v_prefix}/v_value", 0)
        v_values_no_grad = v_values.detach()

        # Compute A_(xi_i) (s, a)
        a_workspace = Workspace()
        a_workspace.set("env/env_obs", 0, states)
        a_workspace.set("action", 0, actions)
        a_network(a_workspace, t=0)
        a_values = a_workspace.get(f"{a_prefix}/a_value", 0)

        # Compute I_i^(s,a) : 1 if V_(phi_i) (s) + A_(xi_i) (s,a) < Q_psi (s, a) else 0
        indicators = (v_values + a_values < q_targets).float()

        # Compute Upsilon_i^a(s) : (1 - rho * I_i^(s,a)) V_(phi_i) (s) + rho * I_i^(s,a) * V_(phi_i)^nograd (s)
        rho = self.gradient_reduction
        upsilon_values = (
            1 - rho * indicators
        ) * v_values + rho * indicators * v_values_no_grad

        # Compute the expression inside Z function
        x = upsilon_values - q_targets
        y = a_values

        # Compute Z(x, y) : (x + y)^2 if x >= 0, else x^2 + y^2
        positive_mask = x >= 0
        z_values = torch.zeros_like(x)
        z_values[positive_mask] = (x[positive_mask] + y[positive_mask]).pow(2)
        z_values[~positive_mask] = x[~positive_mask].pow(2) + y[~positive_mask].pow(2)

        # Compute the final loss
        va_loss = z_values.mean()

        return va_loss

    def _compute_q_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Q-network loss based on the Bellman equation.

        L_Q (psi) = mean [ (Q_psi (s, a) - r - gamma min V_(phi_i^target) (s'))^2 ]
        """

        # Create a workspace to compute Q-values : Q_psi (s, a)
        workspace = Workspace()
        workspace.set("env/env_obs", 0, states)
        workspace.set("action", 0, actions)
        self.q_network(workspace, t=0)
        q_values = workspace.get("q_value", 0)

        # Create workspaces to compute target V-values 1 : V_(phi_1^target) (s')
        target_workspace1 = Workspace()
        target_workspace1.set("env/env_obs", 0, next_states)
        self.v_target_network1(target_workspace1, t=0)
        v1_targets = target_workspace1.get("v1_target/v_value", 0)

        # Create workspaces to compute target V-values 2 : V_(phi_2^target) (s')
        target_workspace2 = Workspace()
        target_workspace2.set("env/env_obs", 0, next_states)
        self.v_target_network2(target_workspace2, t=0)
        v2_targets = target_workspace2.get("v2_target/v_value", 0)

        # Take the minimum of the two target networks : min V_(phi_i^target) (s')
        min_v_targets = torch.min(v1_targets, v2_targets)

        # Comute the mean squared error loss : L_Q (psi)
        targets = rewards + (1.0 - dones) * self.gamma * min_v_targets
        q_loss = torch.mean((q_values - targets.detach()) ** 2)

        return q_loss

    def _compute_policy_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute the policy loss based on the formula.

        L_policy (theta) = mean [ alpha log (policy_theta (a_s | s)) - Q_psi (s, a_s)
        """
        # Create a workspace to compute policy distribution : log policy_theta (a_s | s)
        policy_workspace = Workspace()
        policy_workspace.set("env/env_obs", 0, states)
        self.policy_network(policy_workspace, t=0)

        # Sample actions from the policy
        self.policy_network.sample_action(policy_workspace, t=0)
        self.policy_network.get_log_prob(policy_workspace, t=0)
        actions = policy_workspace.get("action", 0)
        log_probs = policy_workspace.get("log_prob", 0)

        # Create a workspace to compute Q-values for these sampled actions : Q_psi (s, a_s)
        q_workspace = Workspace()
        q_workspace.set("env/env_obs", 0, states)
        q_workspace.set("action", 0, actions)
        self.q_network(q_workspace, t=0)
        q_values = q_workspace.get("q_value", 0)

        # Compute the policy loss : L_policy (theta)
        alpha = self.log_alpha.exp()
        policy_loss = (alpha * log_probs - q_values).mean()

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
            "q_network_state": self.q_network.state_dict(),
            "v_network1_state": self.v_network1.state_dict(),
            "v_network2_state": self.v_network2.state_dict(),
            "v_target_network1_state": self.v_target_network1.state_dict(),
            "v_target_network2_state": self.v_target_network2.state_dict(),
            "a_network1_state": self.a_network1.state_dict(),
            "a_network2_state": self.a_network2.state_dict(),
            "policy_network_state": self.policy_network.state_dict(),
            # Optimizer states
            "q_optimizer_state": self.q_optimizer.state_dict(),
            "v1_optimizer_state": self.v1_optimizer.state_dict(),
            "v2_optimizer_state": self.v2_optimizer.state_dict(),
            "a1_optimizer_state": self.a1_optimizer.state_dict(),
            "a2_optimizer_state": self.a2_optimizer.state_dict(),
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
        self.q_network.load_state_dict(save_dict["q_network_state"])
        self.v_network1.load_state_dict(save_dict["v_network1_state"])
        self.v_network2.load_state_dict(save_dict["v_network2_state"])
        self.v_target_network1.load_state_dict(save_dict["v_target_network1_state"])
        self.v_target_network2.load_state_dict(save_dict["v_target_network2_state"])
        self.a_network1.load_state_dict(save_dict["a_network1_state"])
        self.a_network2.load_state_dict(save_dict["a_network2_state"])
        self.policy_network.load_state_dict(save_dict["policy_network_state"])

        # Restore optimizer states
        self.q_optimizer.load_state_dict(save_dict["q_optimizer_state"])
        self.v1_optimizer.load_state_dict(save_dict["v1_optimizer_state"])
        self.v2_optimizer.load_state_dict(save_dict["v2_optimizer_state"])
        self.a1_optimizer.load_state_dict(save_dict["a1_optimizer_state"])
        self.a2_optimizer.load_state_dict(save_dict["a2_optimizer_state"])
        self.policy_optimizer.load_state_dict(save_dict["policy_optimizer_state"])
        self.alpha_optimizer.load_state_dict(save_dict["alpha_optimizer_state"])

        # Restore temperature parameter
        with torch.no_grad():
            self.log_alpha.copy_(save_dict["log_alpha"])

        # Restore other parameters
        self.params = save_dict["params"]
        self.total_steps = save_dict["total_steps"]

    @classmethod
    def loadagent(cls, path: str) -> "AFU":
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
                "replay_size": 100_000,
                "batch_size": 256,
                "gradient_reduction": 0.5,
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
            "gradient_reduction": ("float", 0.0, 1.0),
            "tau": ("float", 1e-4, 1e-1, True),
            "gamma": ("float", 0.9, 0.999, False),
        }
