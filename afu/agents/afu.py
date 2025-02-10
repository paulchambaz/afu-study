import torch
import gymnasium as gym
import torch.nn as nn
import numpy as np
from bbrl.agents import Agent  # type: ignore
from bbrl_utils.nn import build_mlp  # type: ignore
from bbrl.workspace import Workspace  # type: ignore
from .memory import ReplayBuffer
from tqdm import tqdm  # type: ignore


class ContinuousVFunction(Agent):
    """Value function network that estimates state values V(s)."""

    def __init__(self, state_dim: int, hidden_size: list[int]) -> None:
        super().__init__()

        self.model = build_mlp(
            [state_dim] + hidden_size + [1], activation=nn.ReLU()
        )

    def forward(self, t: int) -> None:
        """Computes value function estimate for the state."""
        obs = self.get(("env/env_obs", t))
        value = self.model(obs).squeeze(-1)
        self.set(("value", t), value)


class GaussianPolicy(Agent):
    """Policy network that outputs a Gaussian distribution over actions."""

    def __init__(
        self,
        state_dim: int,
        hidden_size: list[int],
        action_dim: int,
        log_std_min: float,
        log_std_max: float,
    ) -> None:
        super().__init__()

        self.model = build_mlp(
            [state_dim] + hidden_size + [action_dim], activation=nn.ReLU()
        )

        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, t: int) -> None:
        """Computes mean and log_std of the action distribution."""
        obs = self.get(("env/env_obs", t))
        mean = self.model(obs)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        log_std = log_std.expand_as(mean)

        self.set(("mean", t), mean)
        self.set(("log_std", t), log_std)

    def sample_action(self, workspace: Workspace, t: int) -> None:
        """Samples actions using the reparameterization trick."""
        self.workspace = workspace

        mean = self.get(("mean", t))
        log_std = self.get(("log_std", t))
        std = log_std.exp()

        normal = torch.randn_like(mean)
        action = mean + std * normal
        tanh_action = torch.tanh(action)

        self.set(("sample", t), action)
        self.set(("action", t), tanh_action)


class ContinuousQFunction(Agent):
    """Q-function network that estimates state-action values Q(s,a)."""

    def __init__(
        self, state_dim: int, hidden_size: list[int], action_dim: int, prefix: str
    ) -> None:
        super().__init__()
        self.prefix = prefix

        self.model = build_mlp(
            [state_dim + action_dim] + hidden_size + [1], activation=nn.ReLU()
        )

    def forward(self, t: int) -> None:
        """Computes Q-value estimate for the state-action pair."""
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))

        state_action = torch.cat([obs, action], dim=1)
        q_value = self.model(state_action).squeeze(-1)

        self.set((f"{self.prefix}q_value", t), q_value)


class AFU:
    """Actor Free critic Update implementation."""

    def __init__(self, params: dict) -> None:
        self.params = params

        # Create training environment and validate its observation/action spaces. We need
        # both spaces to have proper shape attributes since we're dealing with continuous
        # state/ation spaces.
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

        self.policy = GaussianPolicy(
            state_dim,
            params["hidden_size"],
            action_dim,
            params["log_std_min"],
            params["log_std_max"],
        )

        self.value = ContinuousVFunction(state_dim, params["hidden_size"])
        self.value_target = ContinuousVFunction(state_dim, params["hidden_size"])

        self.q1 = ContinuousQFunction(
            state_dim, params["hidden_size"], action_dim, prefix="q1/"
        )
        self.q2 = ContinuousQFunction(
            state_dim, params["hidden_size"], action_dim, prefix="q2/"
        )

        self.value_target.load_state_dict(self.value.state_dict())

        self.replay_buffer = ReplayBuffer(params["replay_size"])

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=params["learning_rate"]
        )
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(), lr=params["learning_rate"]
        )
        self.q1_optimizer = torch.optim.Adam(
            self.q1.parameters(), lr=params["learning_rate"]
        )
        self.q2_optimizer = torch.optim.Adam(
            self.q2.parameters(), lr=params["learning_rate"]
        )

        self.total_steps = 0

    def _soft_update(
        self, source_network: nn.Module, target_network: nn.Module
    ) -> None:
        """Performs soft update of target network parameters."""
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
        """Selects an action for the given state."""
        workspace = Workspace()
        state_tensor = torch.FloatTensor(state[None, ...])
        workspace.set("env/env_obs", 0, state_tensor)

        self.policy(workspace, t=0)

        if evaluation:
            action = workspace.get("mean", 0)
        else:
            self.policy.sample_action(workspace, t=0)
            action = workspace.get("action", 0)

        return action.detach().numpy()[0]

    def _compute_critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the critic loss using gradient reduction mechanism."""
        workspace = Workspace()
        workspace.set("env/env_obs", 0, next_states)
        self.value_target(workspace, t=0)
        target_values = workspace.get("value", 0)

        target_q = (
            rewards
            + (1.0 - dones) * self.params["gamma"] * target_values.detach()
        )

        workspace.set("env/env_obs", 0, states)
        self.value(workspace, t=0)
        current_values = workspace.get("value", 0)

        workspace.set("action", 0, actions)
        self.q1(workspace, t=0)
        self.q2(workspace, t=0)
        q1_values = workspace.get("q1/q_value", 0)
        q2_values = workspace.get("q2/q_value", 0)

        advantages1 = q1_values - current_values.detach()
        advantages2 = q2_values - current_values.detach()

        grad_red = self.params["gradient_reduction"]
        mask = (target_q <= current_values).float().detach()
        reduced_values = mask * current_values + (1 - mask) * (
            grad_red * current_values + (1 - grad_red) * current_values.detach()
        )


        critic_loss1 = (advantages1**2 + (reduced_values - target_q)**2).mean()
        critic_loss2 = (advantages2**2 + (reduced_values - target_q)**2).mean()
        
        return critic_loss1, critic_loss2

    def _compute_actor_loss(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the actor loss and entropy terms."""
        workspace = Workspace()
        workspace.set("env/env_obs", 0, states)

        self.policy(workspace, t=0)
        self.policy.sample_action(workspace, t=0)
        actions = workspace.get("action", 0)

        workspace.set("action", 0, actions)
        self.q1(workspace, t=0)
        self.q2(workspace, t=0)
        q1_values = workspace.get("q1/q_value", 0)
        q2_values = workspace.get("q2/q_value", 0)

        q_values = torch.min(q1_values, q2_values)

        policy_loss = -q_values.mean()

        # we return a zero entropy term for simplicity, might need to update it later
        return policy_loss, torch.zeros_like(policy_loss)

    def _compute_value_loss(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the value function loss."""
        workspace = Workspace()
        workspace.set("env/env_obs", 0, states)

        self.value(workspace, t=0)
        current_values = workspace.get("value", 0)

        self.policy(workspace, t=0)
        self.policy.sample_action(workspace, t=0)
        actions = workspace.get("action", 0)

        workspace.set("action", 0, actions)
        self.q1(workspace, t=0)
        self.q2(workspace, t=0)
        q1_values = workspace.get("q1/q_value", 0)
        q2_values = workspace.get("q2/q_value", 0)

        q_values = torch.min(q1_values, q2_values)

        value_loss = nn.MSELoss()(current_values, q_values.detach())

        return value_loss

    def update(self) -> tuple[float, float, float, float]:
        """Performs a single update step of all networks."""
        if len(self.replay_buffer) < self.params["batch_size"]:
            return 0.0, 0.0, 0.0, 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.params["batch_size"],
            continuous=True,
        )

        q1_loss, q2_loss = self._compute_critic_loss(
            states, actions, rewards, next_states, dones
        )
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        value_loss = self._compute_value_loss(states, next_states, rewards, dones)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        actor_loss, _ = self._compute_actor_loss(states)
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

        self._soft_update(self.value, self.value_target)

        return (
            actor_loss.item(),
            q1_loss.item(),
            q2_loss.item(),
            value_loss.item(),
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
        """Saves model parameters to a file."""
        save_dict = {
            "policy_state": self.policy.state_dict(),
            "value_state": self.value.state_dict(),
            "value_target_state": self.value_target.state_dict(),
            "q1_state": self.q1.state_dict(),
            "q2_state": self.q2.state_dict(),
            "policy_optimizer_state": self.policy_optimizer.state_dict(),
            "value_optimizer_state": self.value_optimizer.state_dict(),
            "q1_optimizer_state": self.q1_optimizer.state_dict(),
            "q2_optimizer_state": self.q2_optimizer.state_dict(),
            "params": self.params,
            "total_steps": self.total_steps,
        }
        torch.save(save_dict, path)

    def load(self, path: str) -> None:
        """Loads model parameters from a file."""
        save_dict = torch.load(path)

        self.policy.load_state_dict(save_dict["policy_state"])
        self.value.load_state_dict(save_dict["value_state"])
        self.value_target.load_state_dict(save_dict["value_target_state"])
        self.q1.load_state_dict(save_dict["q1_state"])
        self.q2.load_state_dict(save_dict["q2_state"])

        self.policy_optimizer.load_state_dict(save_dict["policy_optimizer_state"])
        self.value_optimizer.load_state_dict(save_dict["value_optimizer_state"])
        self.q1_optimizer.load_state_dict(save_dict["q1_optimizer_state"])
        self.q2_optimizer.load_state_dict(save_dict["q2_optimizer_state"])

        self.params = save_dict["params"]
        self.total_steps = save_dict["total_steps"]

    @classmethod
    def load_agent(cls, path: str) -> "AFU":
        """Creates a new agent instance from saved parameters."""
        save_dict = torch.load(path)
        agent = cls(save_dict["params"])
        agent.load(path)
        return agent
