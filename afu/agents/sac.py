import torch
import gymnasium as gym
import torch.nn as nn
import numpy as np
from bbrl.agents import Agent  # type: ignore
from bbrl_utils.nn import build_mlp  # type: ignore
from bbrl.workspace import Workspace  # type: ignore
from .memory import ReplayBuffer
from tqdm import tqdm  # type: ignore


class GaussianPolicy(Agent):
    """A neural network that outputs a Gaussian distribution over actions."""

    def __init__(
        self, state_dim: int, hidden_size: list[int], action_dim: int
    ) -> None:
        """Initialize gaussian policy network with given dimensions."""
        super().__init__()

        self.model = build_mlp(
            [state_dim] + hidden_size + [action_dim],
            activation=nn.ReLU(),
        )

        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, t: int) -> None:
        """Compute mean and log_std of action distribution for a given state."""
        obs = self.get(("env/env_obs", t))
        mean = self.model(obs)
        log_std = self.log_std.expand_as(mean)

        self.set(("mean", t), mean)
        self.set(("log_std", t), log_std)

    def sample_action(self, workspace: Workspace, t: int) -> None:
        """Sample action from the Gaussian distribution using reparameterization trick."""
        self.workspace = workspace

        mean = self.get(("mean", t))
        log_std = self.get(("log_std", t))
        std = log_std.exp()

        normal = torch.randn_like(mean)
        action = mean + std * normal
        tanh_action = torch.tanh(action)

        self.set(("sample", t), action)
        self.set(("action", t), tanh_action)

    def get_log_prob(self, workspace: Workspace, t: int) -> None:
        """Compute log probability of an action under the Gaussian distribution."""
        self.workspace = workspace

        mean = self.get(("mean", t))
        log_std = self.get(("log_std", t))
        sample = self.get(("sample", t))
        action = self.get(("action", t))

        std = log_std.exp()
        normal_log_prob = (-0.5 * ((sample - mean) / std).pow(2) - log_std).sum(
            dim=-1
        )
        log_prob = normal_log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(
            dim=-1
        )

        self.set(("log_prob", t), log_prob)


class SoftQNetwork(Agent):
    """Q-network that estimates state-action values."""

    def __init__(
        self, state_dim: int, hidden_size: list[int], action_dim: int, prefix: str
    ) -> None:
        super().__init__()
        self.prefix = prefix

        self.model = build_mlp(
            [state_dim + action_dim] + hidden_size + [1], activation=nn.ReLU()
        )

    def forward(self, t: int) -> None:
        """Compute Q-value for a given state-action pair."""
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))

        state_action = torch.cat([obs, action], dim=1)
        q_value = self.model(state_action).squeeze(-1)

        self.set((f"{self.prefix}q_value", t), q_value)


class SAC:
    """Soft Actor-Critic implementation."""

    def __init__(self, params: dict) -> None:
        """Initialize networks, optimizers and other components."""
        self.params = params

        # Create training environment and validate its observation/action spaces.
        # We need both spaces to have proper shape attributes since we're dealing
        # with continuous state/action spaces.
        self.train_env = gym.make(params["env_name"])
        if (
            not hasattr(self.train_env.observation_space, "shape")
            or self.train_env.observation_space.shape is None
        ):
            raise ValueError(
                "Environment's observation space must have a shape attribute"
            )
        state_dim = self.train_env.observation_space.shape[0]

        if (
            not hasattr(self.train_env.action_space, "shape")
            or self.train_env.action_space.shape is None
        ):
            raise ValueError(
                "Environment's action space must have a shape attribute"
            )
        action_dim = self.train_env.action_space.shape[0]
        self.action_dim = action_dim

        self.policy = GaussianPolicy(
            state_dim, params["policy_hidden_size"], action_dim
        )

        self.q1 = SoftQNetwork(
            state_dim, params["q_hidden_size"], action_dim, prefix="q1/"
        )
        self.q2 = SoftQNetwork(
            state_dim, params["q_hidden_size"], action_dim, prefix="q2/"
        )

        self.target_q1 = SoftQNetwork(
            state_dim, params["q_hidden_size"], action_dim, prefix="target_q1/"
        )
        self.target_q2 = SoftQNetwork(
            state_dim, params["q_hidden_size"], action_dim, prefix="target_q2/"
        )

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.log_alpha = torch.zeros(1, requires_grad=True)

        self.replay_buffer = ReplayBuffer(params["replay_size"])

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=params["policy_lr"]
        )
        self.q1_optimizer = torch.optim.Adam(
            self.q1.parameters(), lr=params["q_lr"]
        )
        self.q2_optimizer = torch.optim.Adam(
            self.q2.parameters(), lr=params["q_lr"]
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=params["alpha_lr"]
        )

        self.total_steps = 0

    def _soft_update(
        self, source_network: nn.Module, target_network: nn.Module
    ) -> None:
        """Perform soft update of target network parameters."""
        for target_param, source_param in zip(
            target_network.parameters(), source_network.parameters()
        ):
            target_param.data.copy_(
                (1 - self.params["tau"]) * target_param.data
                + self.params["tau"] * source_param.data
            )

    def select_action(
        self, state: np.ndarray, evaluation: bool = False
    ) -> np.ndarray:
        """Sample action from policy, optionally without exploring for evaluation."""
        workspace = Workspace()
        state_tensor = torch.FloatTensor(state)
        workspace.set("env/env_obs", 0, state_tensor)

        self.policy(workspace, t=0)

        if evaluation:
            action = workspace.get("mean", 0)
        else:
            self.policy.sample_action(workspace, t=0)
            action = workspace.get("action", 0)

        return action.detach().numpy()

    def _compute_q_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute loss for both Q-networks."""
        targets = self._compute_targets(rewards, next_states, dones)

        workspace = Workspace()
        workspace.set("env/env_obs", 0, states)
        workspace.set("action", 0, actions)

        self.q1(workspace, t=0)
        self.q2(workspace, t=0)
        current_q1 = workspace.get("q1/q_value", 0)
        current_q2 = workspace.get("q2/q_value", 0)

        q1_loss = nn.MSELoss()(current_q1, targets.detach())
        q2_loss = nn.MSELoss()(current_q2, targets.detach())

        return q1_loss, q2_loss

    def _compute_policy_loss(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute policy loss including entropy term."""
        workspace = Workspace()
        workspace.set("env/env_obs", 0, states)

        self.policy(workspace, t=0)
        self.policy.sample_action(workspace, t=0)
        self.policy.get_log_prob(workspace, t=0)

        actions = workspace.get("action", 0)
        log_probs = workspace.get("log_prob", 0)

        workspace.set("action", 0, actions)
        self.q1(workspace, t=0)
        self.q2(workspace, t=0)
        q1_values = workspace.get("q1/q_value", 0)
        q2_values = workspace.get("q2/q_value", 0)
        q_values = torch.min(q1_values, q2_values)

        alpha = self.log_alpha.exp()
        policy_loss = (alpha * log_probs - q_values).mean()

        return policy_loss, log_probs

    def _compute_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute target Q-values including entropy term"""
        workspace = Workspace()

        workspace.set("env/env_obs", 0, next_states)
        self.policy(workspace, t=0)
        self.policy.sample_action(workspace, t=0)
        self.policy.get_log_prob(workspace, t=0)
        next_actions = workspace.get("action", 0)
        next_log_probs = workspace.get("log_prob", 0)

        workspace.set("action", 0, next_actions)
        self.target_q1(workspace, t=0)
        self.target_q2(workspace, t=0)
        next_q1 = workspace.get("target_q1/q_value", 0)
        next_q2 = workspace.get("target_q2/q_value", 0)

        next_q = torch.min(next_q1, next_q2)

        alpha = self.log_alpha.exp()
        next_q = next_q - alpha * next_log_probs

        targets = rewards + (1 - dones) * self.params["gamma"] * next_q

        return targets

    def _adjust_temperature(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Update temperature parameter to maintain target entropy."""
        target_entropy = -self.action_dim

        alpha = self.log_alpha.exp()
        alpha_loss = -(
            self.log_alpha * (log_probs + target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return alpha_loss

    def update(self) -> tuple[float, float, float, float]:
        """Perform one update setp on all networks."""
        if len(self.replay_buffer) < self.params["batch_size"]:
            return 0.0, 0.0, 0.0, 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.params["batch_size"],
            continuous=True,
        )

        q1_loss, q2_loss = self._compute_q_loss(
            states, actions, rewards, next_states, dones
        )
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        policy_loss, log_porbs = self._compute_policy_loss(states)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha_loss = self._adjust_temperature(log_porbs)

        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)

        return (
            policy_loss.item(),
            q1_loss.item(),
            q2_loss.item(),
            alpha_loss.item(),
        )

    def train(self) -> dict:
        """Train the agent over multiple episodes."""
        episode_rewards = []
        progress = tqdm(range(self.params["max_episodes"]), desc="Training")

        for episode in progress:
            state, _ = self.train_env.reset()
            episode_reward = 0.0

            for step in range(self.params["max_steps"]):
                action = self.select_action(state)
                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = self.train_env.step(action)
                done = terminated or truncated

                self.replay_buffer.push(state, action, reward, next_state, done)
                _, _, _, _ = self.update()

                state = next_state
                episode_reward += float(reward)
                self.total_steps += 1

                if done:
                    break

            episode_rewards.append(episode_reward)

            if len(episode_rewards) >= 10:
                avg_reward = np.mean(episode_rewards[-10:])
                progress.set_postfix(
                    {"avg_reward": f"{avg_reward:.2f}"}, refresh=True
                )

        return {"episode_rewards": episode_rewards}

    def save(self, path: str) -> None:
        save_dict = {
            "policy_state": self.policy.state_dict(),
            "q1_state": self.q1.state_dict(),
            "q2_state": self.q2.state_dict(),
            "target_q1_state": self.target_q1.state_dict(),
            "target_q2_state": self.target_q2.state_dict(),
            "policy_optimizer_state": self.policy_optimizer.state_dict(),
            "q1_optimizer_state": self.q1_optimizer.state_dict(),
            "q2_optimizer_state": self.q2_optimizer.state_dict(),
            "alpha_optimizer_state": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha,
            "params": self.params,
            "total_steps": self.total_steps,
        }
        torch.save(save_dict, path)

    def load(self, path: str) -> None:
        save_dict = torch.load(path)

        self.policy.load_state_dict(save_dict["policy_state"])
        self.q1.load_state_dict(save_dict["q1_state"])
        self.q2.load_state_dict(save_dict["q2_state"])
        self.target_q1.load_state_dict(save_dict["target_q1_state"])
        self.target_q2.load_state_dict(save_dict["target_q2_state"])

        self.policy_optimizer.load_state_dict(save_dict["policy_optimizer_state"])
        self.q1_optimizer.load_state_dict(save_dict["q1_optimizer_state"])
        self.q2_optimizer.load_state_dict(save_dict["q2_optimizer_state"])
        self.alpha_optimizer.load_state_dict(save_dict["alpha_optimizer_state"])

        self.log_alpha = save_dict["log_alpha"]
        self.params = save_dict["params"]
        self.total_steps = save_dict["total_steps"]

    @classmethod
    def load_agent(cls, path: str) -> "SAC":
        save_dict = torch.load(path)
        agent = cls(save_dict["params"])
        agent.load(path)
        return agent
