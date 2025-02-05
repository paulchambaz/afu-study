import torch
import gymnasium as gym
import torch.nn as nn
import numpy as np
from bbrl.agents import Agent  # type: ignore
from bbrl_utils.nn import build_mlp  # type: ignore
from bbrl.workspace import Workspace  # type: ignore
from .memory import ReplayBuffer


class Actor(Agent):
    def __init__(
        self, state_dim: int, hidden_size: list[int], action_dim: int
    ) -> None:
        super().__init__()

        self.model = build_mlp(
            [state_dim] + hidden_size + [action_dim],
            activation=nn.ReLU(),
            output_activation=nn.Tanh(),
        )

    def forward(self, t: int) -> None:
        obs = self.get(("env/env_obs", t))
        action = self.model(obs)
        self.set(("action", t), action)


class Critic(Agent):
    def __init__(
        self, state_dim: int, hidden_size: list[int], action_dim: int
    ) -> None:
        super().__init__()

        self.model = build_mlp(
            [state_dim + action_dim] + hidden_size + [1], activation=nn.ReLU()
        )

    def forward(self, t: int) -> None:
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))

        state_action = torch.cat([obs, action], dim=1)
        q_value = self.model(state_action).squeeze(-1)

        self.set((f"{self.prefix}q_value", t), q_value)


class GaussianNoise:
    def __init__(self, action_dimension: int, noise_std: float) -> None:
        self.action_dimension = action_dimension
        self.noise_std = noise_std
        self.reset()

    def sample(self) -> np.ndarray:
        noise = np.random.normal(0, self.noise_std, size=self.action_dimension)
        return noise

    def reset(self) -> None:
        pass


class DDPG:
    def __init__(self, params: dict) -> None:
        self.params = params

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

        self.actor = Actor(state_dim, params["actor_hidden_size"], action_dim)
        self.target_actor = Actor(
            state_dim, params["actor_hidden_size"], action_dim
        )
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, params["critic_hidden_size"], action_dim)
        self.target_critic = Critic(
            state_dim, params["critic_hidden_size"], action_dim
        ).with_prefix("target_critic/")
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.noise = GaussianNoise(action_dim, params["noise_std"])
        self.replay_buffer = ReplayBuffer(params["replay_size"])

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=params["actor_lr"]
        )
        self.critic_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=params["critic_lr"]
        )

        self.total_steps = 0

    def _soft_update(
        self, source_network: nn.Module, target_network: nn.Module
    ) -> None:
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
        workspace = Workspace()
        workspace.set("env/env_obs", 0, torch.FloatTensor([state]))

        self.actor(workspace, t=0)
        action = workspace.get("action", 0).detach().numpy()

        if not evaluation:
            action += self.noise.sample()
            # TODO: verify this is acceptable - may depend on the problem - maybe this should be a hyperparameter
            action = np.clip(action, -1, 1)

        return action

    def _compute_critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        workspace = Workspace()

        workspace.set("env/env_obs", 0, next_states)
        self.target_actor(workspace, t=0)
        next_actions = workspace.get("action", 0)

        workspace.set("action", 0, next_actions)
        self.target_critic(workspace, t=0)
        next_q_values = workspace.get("target_critic/q_value", 0)

        target_q = (
            rewards + (1 - dones) * self.params["gamma"] * next_q_values.detach()
        )

        workspace.set("env/env_obs", 0, states)
        workspace.set("action", 0, actions)
        self.critic(workspace, t=0)
        current_q = workspace.get("critic/q_value", 0)

        return nn.MSELoss()(current_q, target_q)

    def _compute_actor_loss(self, states: torch.Tensor) -> torch.Tensor:
        workspace = Workspace()
        workspace.set("env/env_obs", 0, states)

        self.actor(workspace, t=0)
        actions = workspace.get("action", 0)

        workspace.set("action", 0, actions)
        self.critic(workspace, t=0)
        q_values = workspace.get("critic/q_value", 0)

        return -q_values.mean()

    def update(self) -> tuple[float, float]:
        if len(self.replay_buffer) < self.params["batch_size"]:
            return 0.0, 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.params["batch_size"],
            True,
        )

        critic_loss = self._compute_critic_loss(
            states, actions, rewards, next_states, dones
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = self._compute_actor_loss(states)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.critic, self.target_critic)
        self._soft_update(self.actor, self.target_actor)

        return actor_loss.item(), critic_loss.item()

    def train(self) -> dict:
        episode_rewards = []

        for episode in range(self.params["max_episodes"]):
            state, _ = self.train_env.reset()
            episode_reward = 0.0
            self.noise.reset()

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
                _, _ = self.update()

                state = next_state
                episode_reward += float(reward)
                self.total_steps += 1

                if done:
                    break

            episode_rewards.append(episode_reward)

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")

        return {"episode_rewards": episode_rewards}

    def save(self, path: str) -> None:
        save_dict = {
            "actor_state": self.actor.state_dict(),
            "critic_state": self.critic.state_dict(),
            "target_actor_state": self.target_actor.state_dict(),
            "target_critic_state": self.target_critic.state_dict(),
            "actor_optimizer_state": self.actor_optimizer.state_dict(),
            "critic_optimizer_state": self.critic_optimizer.state_dict(),
            "params": self.params,
            "total_steps": self.total_steps,
        }
        torch.save(save_dict, path)

    def load(self, path: str) -> None:
        save_dict = torch.load(path)

        self.actor.load_state_dict(save_dict["actor_state"])
        self.critic.load_state_dict(save_dict["critic_state"])
        self.target_actor.load_state_dict(save_dict["target_actor_state"])
        self.target_critic.load_state_dict(save_dict["target_critic_state"])
        self.actor_optimizer.load_state_dict(save_dict["actor_optimizer_state"])
        self.critic_optimizer.load_state_dict(save_dict["critic_optimizer_state"])

        self.params = save_dict["params"]
        self.total_steps = save_dict["total_steps"]

    @classmethod
    def load_agent(cls, path: str) -> "DDPG":
        save_dict = torch.load(path)
        agent = cls(save_dict["params"])
        agent.load(path)
        return agent
