import torch
import gymnasium as gym
import torch.nn as nn
import numpy as np
from bbrl.agents import Agent  # type: ignore
from bbrl_utils.nn import build_mlp  # type: ignore
from bbrl.workspace import Workspace  # type: ignore
from .memory import ReplayBuffer
from tqdm import tqdm  # type: ignore


class Actor(Agent):
    """A neural network that maps states to continuous actions in a deterministic policy"""

    def __init__(
        self, state_dim: int, hidden_size: list[int], action_dim: int
    ) -> None:
        """Initialize actor network with given dimensions."""
        # The actor serves as the policy network in DDPG, directly outputting the best
        # predicted action for any given state. It uses a tanh activation on the output
        # layer to bound actions to [-1, 1], which can then be scaled to the
        # environment's action range.
        super().__init__()

        # Build neural network that maps states to actions. ReLU activations maintain
        # good gradient flow in hidden layers, while tanh ensures actins are bounded
        # [-1, 1].
        self.model = build_mlp(
            [state_dim] + hidden_size + [action_dim],
            activation=nn.ReLU(),
            output_activation=nn.Tanh(),
        )

    def forward(self, t: int) -> None:
        """Compute continuous action for a given state observation"""
        # Gets states observation from workspace, passes it through the network to
        # generate an action, and stores the action back in the workspace for the critic
        # to use.
        obs = self.get(("env/env_obs", t))
        action = self.model(obs)
        self.set(("action", t), action)


class Critic(Agent):
    """A neural network that estimates Q-values for state-action pairs."""

    def __init__(
        self,
        state_dim: int,
        hidden_size: list[int],
        action_dim: int,
        prefix: str,
    ) -> None:
        """Initialize critic network with given dimensions."""
        # The critic evaluates the quality of actions in different states by estimating
        # their Q-values. It takes both state and action as input to predict the expected
        # cumulative reward for taking that action in that state.
        super().__init__()
        self.prefix = prefix

        self.model = build_mlp(
            [state_dim + action_dim] + hidden_size + [1], activation=nn.ReLU()
        )

    def forward(self, t: int) -> None:
        """Compute Q-value for a given state-action pair."""
        # Gets state and action from workspace, concatenates them into a single input,
        # passes through network to estimate Q-value, and stores result in workspace.
        # The squeeze removes the last dimension as Q-values are scalar per state-action.
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))

        state_action = torch.cat([obs, action], dim=1)
        q_value = self.model(state_action).squeeze(-1)

        self.set((f"{self.prefix}q_value", t), q_value)


class GaussianNoise:
    """Gaussian noise generator for continuous action space exploration."""

    def __init__(self, action_dimension: int, noise_std: float) -> None:
        """Initialize noise generator with given parameters."""

        # Adds random normal noise to actions during training to encourage exploration of
        # the continuous action space. Unlike epsilon-greedy which works for discrete
        # actions, Gaussian noise provides smooth exploration around the current action.
        self.action_dimension = action_dimension
        self.noise_std = noise_std
        self.reset()

    def sample(self) -> np.ndarray:
        """Generate random Gaussian noise for action exploration."""
        noise = np.random.normal(0, self.noise_std, size=self.action_dimension)
        return noise

    def reset(self) -> None:
        """Reset noise state for new episode."""
        self.noise = self.sample()


class DDPG:
    """Deep Deterministic Policy Gradient implementation for continuous action spaces."""

    def __init__(self, params: dict) -> None:
        """Initialize DDPG agent with training parameters."""
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

        # Inititialize the core networks, actor (policy) network and its target for
        # stable learning and critic (value network and its target for stable learning.
        # Target networks start as copies of their original networks.
        self.actor = Actor(state_dim, params["actor_hidden_size"], action_dim)
        self.target_actor = Actor(
            state_dim, params["actor_hidden_size"], action_dim
        )
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic = Critic(
            state_dim, params["critic_hidden_size"], action_dim, prefix="critic/"
        )
        self.target_critic = Critic(
            state_dim,
            params["critic_hidden_size"],
            action_dim,
            prefix="target_critic/",
        )
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Exploration noise helps explore continuous action space.
        self.noise = GaussianNoise(action_dim, params["noise_std"])

        # Replay buffer stores transitions for stable learning.
        self.replay_buffer = ReplayBuffer(params["replay_size"])

        # Seperate optimizer allow different learning rates for actor and critic
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=params["actor_lr"]
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=params["critic_lr"]
        )

        self.total_steps = 0

    def _soft_update(
        self, source_network: nn.Module, target_network: nn.Module
    ) -> None:
        """Slowly update target network parameters from source network."""
        # Perform soft update, which creates a slowly-moving average of the source
        # network parameters
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
        """Choose action for given state, optionally adding exploration noise."""
        # Convert state for tensor and use workspace foor BBRL compability. Get action
        # from actor network, add exploration noise during training and clip actions to
        # valid range to prevent extreme outputs.
        workspace = Workspace()
        state_tensor = torch.FloatTensor(state)
        workspace.set("env/env_obs", 0, state_tensor)

        self.actor(workspace, t=0)
        action = workspace.get("action", 0).detach().numpy()

        if not evaluation:
            action += self.noise.sample()
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
        """Compute MSE loss for critic network using temporal difference learning."""

        # First compute target Q-values using target networks: 1. Get next actions from
        # target actor (what actions would we take next?). 2. Get their Q-values from
        # target critic (how good would those actions be?). 3. Compute target Q using
        # Bellman equation.
        workspace = Workspace()
        workspace.set("env/env_obs", 0, next_states)
        self.target_actor(workspace, t=0)
        next_actions = workspace.get("action", 0)

        workspace.set("action", 0, next_actions)
        self.target_critic(workspace, t=0)
        next_q_values = workspace.get("target_critic/q_value", 0)

        # Calculate target Q-values using Bellman equation. For terminal states (done=1),
        # ony use immediate reward. Otherwise include discounted future value from
        # target networks.
        target_q = (
            rewards + (1 - dones) * self.params["gamma"] * next_q_values.detach()
        )

        # Now compute current Q-values from our critic network. These represent our
        # current estimates that we want to improve.
        workspace.set("env/env_obs", 0, states)
        workspace.set("action", 0, actions)
        self.critic(workspace, t=0)
        current_q = workspace.get(f"{self.critic.prefix}q_value", 0)

        # Return MSE between current and target Q-values. This tells us how wrong our
        # critics's perdictions were.
        return nn.MSELoss()(current_q, target_q)

    def _compute_actor_loss(self, states: torch.Tensor) -> torch.Tensor:
        """Compute loss for actor network by maximizing critic's Q-values."""

        # Use current actor to generate actions for states. Then get Q-values for these
        # state-action pairs from critic. Taking negative mean converts maximization to
        # minimization. This encourages actor to choose actions that critic rates highly.
        workspace = Workspace()
        workspace.set("env/env_obs", 0, states)

        self.actor(workspace, t=0)
        actions = workspace.get("action", 0)

        workspace.set("action", 0, actions)
        self.critic(workspace, t=0)
        q_values = workspace.get("critic/q_value", 0)

        return -q_values.mean()

    def update(self) -> tuple[float, float]:
        """Perform one update step on critic and actor networks."""

        # Skip update if we don't have enough samples for a full batch
        if len(self.replay_buffer) < self.params["batch_size"]:
            return 0.0, 0.0

        # Sample a batch of transitions from replay buffer. These will be used to update
        # both networks.
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.params["batch_size"],
            continuous=True,
        )

        # First update critic by minimizing TD error. Clear gradients, compute loss,
        # update weights.
        critic_loss = self._compute_critic_loss(
            states, actions, rewards, next_states, dones
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Then update actor by maximizing Q-values. Clear gradients, compute loss,
        # update weights.
        actor_loss = self._compute_actor_loss(states)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Finally, soft update target networks to track main networks.
        self._soft_update(self.critic, self.target_critic)
        self._soft_update(self.actor, self.target_actor)

        return actor_loss.item(), critic_loss.item()

    def train(self) -> dict:
        """Train the agent over multiple episodes."""

        # Track rewards for performance monitoring
        episode_rewards = []
        progress = tqdm(range(self.params["max_episodes"]), desc="Training")

        # Each episode is one complete run through the environment
        for episode in progress:
            state, _ = self.train_env.reset()
            episode_reward = 0.0
            self.noise.reset()

            # Step through episode until done or max steps reached
            for step in range(self.params["max_steps"]):
                # Core loop: select action, execute it, store transition, update
                action = self.select_action(state)
                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = self.train_env.step(action)
                done = terminated or truncated

                # Store transition and update networks
                self.replay_buffer.push(state, action, reward, next_state, done)
                _, _ = self.update()

                state = next_state
                episode_reward += float(reward)
                self.total_steps += 1

                if done:
                    break

            # Store episode reward and periodically print progress
            episode_rewards.append(episode_reward)

            if len(episode_rewards) >= 10:
                avg_reward = np.mean(episode_rewards[-10:])
                progress.set_postfix(
                    {"avg_reward": f"{avg_reward:.2f}"}, refresh=True
                )

        return {"episode_rewards": episode_rewards}

    def save(self, path: str) -> None:
        # Save everything needed to restore the agent: network weights,
        # optimizer state, parameters and training progress
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

        # Load the saved dictionary and restore each component
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
        """Create a new DQN agent from a saved file."""

        # First create a new agent, then load the saved state into it
        save_dict = torch.load(path)
        agent = cls(save_dict["params"])
        agent.load(path)
        return agent
