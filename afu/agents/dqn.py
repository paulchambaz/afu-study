import torch
from omegaconf import OmegaConf
import gymnasium as gym
import torch.nn as nn
import random
import numpy as np
from bbrl_utils.nn import build_mlp  # type: ignore
from bbrl.workspace import Workspace  # type: ignore
from bbrl.agents import TimeAgent, SeedableAgent, SerializableAgent

# from bbrl.utils.replay_buffer import ReplayBuffer
from .memory import ReplayBuffer


class DiscreteQNetwork(TimeAgent, SeedableAgent, SerializableAgent):
    """Q-Network that estimates"""

    def __init__(
        self, state_dim: int, hidden_dims: list[int], action_dim: int, name="critic"
    ):
        super().__init__(name=name)

        self.model = build_mlp(
            [state_dim] + hidden_dims + [action_dim], activation=nn.ReLU()
        )

    def forward(self, t: int, choose_action: bool = True) -> None:
        """Compute Q-value for a given state."""
        # get state from workspace
        obs = self.get(("env/env_obs", t))

        # compute Q-value
        q_values = self.model(obs)
        self.set((f"{self.name}/q_values", t), q_values)


class DQN:
    def __init__(self, hyperparameters: OmegaConf):
        self.params = hyperparameters

        train_env = gym.make(self.params.env_name)

        state_dim = train_env.observation_space.shape[0]
        action_dim = train_env.action_space.n

        hidden_dims = [self.params.hidden_size, self.params.hidden_size]

        self.q_network = DiscreteQNetwork(
            state_dim=state_dim,
            hidden_dims=hidden_dims,
            action_dim=action_dim,
            name="q",
        )

        self.q_target_network = DiscreteQNetwork(
            state_dim=state_dim,
            hidden_dims=hidden_dims,
            action_dim=action_dim,
            name="q_target",
        )

        self.q_target_network.load_state_dict(self.q_network.state_dict())

        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.params.q_lr
        )

        # self.replay_buffer = ReplayBuffer(max_size=self.params.replay_size)
        self.replay_buffer = ReplayBuffer(self.params.replay_size)
        self.batch_size = self.params.batch_size
        self.gamma = self.params.gamma
        self.decay_rate = self.params.decay_rate
        self.target_update = self.params.target_update
        self.epsilon = 0.6

    def select_action(self, state: np.ndarray, evaluation: bool = False) -> np.ndarray:
        workspace = Workspace()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        workspace.set("env/env_obs", 0, state_tensor)

        self.q_network(workspace, t=0)
        q_values = workspace.get("q/q_values", t=0)

        self.epsilon = 0.0 if evaluation else self.epsilon * 0.9999

        if random.random() > self.epsilon:
            action = q_values.argmax().item()
        else:
            action = int(round(random.random()))

        return action

    def update(self):
        """Update network based on sampled experience from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return

        rb_workspace = self.replay_buffer.get_shuffled(self.batch_size)
        states, actions, rewards, next_states, dones = rb_workspace[
            "env/env_obs"[0],
            "action"[0],
            "env/reward"[1],
            "env_env_obs"[1],
            "env/dones"[1],
        ]

        # Update Q-network
        q_loss = self._compute_q_loss(states, actions, rewards, next_states, dones)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Periodically copy Q-network weights to target network
        if self.total_steps % self.target_update == 0:
            self.q_target_network.load_state_dict(self.q_network.state_dict())

    def _compute_q_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Q-network loss bsaed on the Bellman equation.

        L_Q (theta) = mean [ (Q_theta (s, a) - r - gamma * max [ Q_theta^target (s', a_s) ])^2 ]
        """
        # Compute Q-values : Q_theta (s, a)
        workspace = Workspace()
        workspace.set("env/env_obs", 0, states)
        self.q_network(workspace, t=0, choose_action=False)
        q_values = workspace.get("q/q_values", 0)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target max of Q-values : max [ Q_theta^target (s', a_s) ]
        workspace.set("env/env_obs", 0, next_states)
        self.target_network(workspace, t=0, choose_action=False)
        next_q_values = workspace.get("target_q/q_values", 0)
        max_next_q_values = next_q_values.max(1)[0].detach()

        # Compute the targets : r + gamma * max [ Q_theta^target (s', a_s) ]
        q_targets = rewards + (1.0 - dones) * self.gamma * max_next_q_values

        # Compute the mean squared error loss
        q_loss = 0.5 * torch.mean((q_values - q_targets.detach()) ** 2)

        return q_loss

    def get_weights(self) -> dict:
        return {
            # Network states
            "q_network_state": self.q_network.state_dict(),
            "q_target_network_state": self.q_target_network.state_dict(),
            # Optimizer states
            "q_optimizer_state": self.q_optimizer,
            # Other parameters
            "epsilon": self.epsilon,
            "params": self.params,
        }

    def save(self, path: str) -> None:
        save_dict = self.get_weights()
        torch.save(save_dict, path)

    def load(self, path: str) -> None:
        save_dict = torch.load(path)

        # Resore network states
        self.q_network.load_state_dict(save_dict["q_network_state"])
        self.q_target_network.load_state_dict(save_dict["q_target_network_state"])

        # Restore optimizer states
        self.q_optimizer.load_state_dict(save_dict["q_optimizer_state"])

        # Restore other parameters
        self.epsilon = save_dict["epsilon"]
        self.params = save_dict["params"]

    @classmethod
    def loadagent(cls, path: str) -> "DQN":
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
                "replay_size": 100_000,
                "batch_size": 256,
                "gamma": 0.99,
                "decay_rate": 0.9999,
                "target_update": 512,
            }
        )
