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
    """Value function network that estimates state values V(S)."""

    def __init__(
        self, state_dim: int, hidden_size: list[int], prefix: str
    ) -> None:
        """Initialize value function network.

        * state_dim: dimension of the state space
        * hidden_size: list of hidden layer sizes
        """
        super().__init__()
        self.prefix = prefix

        self.model = build_mlp(
            [state_dim] + hidden_size + [1], activation=nn.ReLU()
        )

    def forward(self, t: int) -> None:
        """Compute value function estimates for the state.

        * t: time step
        """
        obs = self.get(("env/env_obs", t))
        value = self.model(obs).squeeze(-1)
        self.set((f"{self.prefix}value", t), value)


class ContiniousQFunction(Agent):
    """Q-function network that estimates state-action values Q(s, a)."""

    def __init__(
        self, state_dim: int, action_dim: int, hidden_size: list[int], prefix: str
    ) -> None:
        """Initialize Q-function network.

        * state_dim: dimension of the state space
        * action_dim: dimension of the action space
        * hidden_size: list of hidden layer sizes
        * prefix: prefix for workspace variables
        """
        super().__init__()
        self.prefix = prefix

        self.model = build_mlp(
            [state_dim + action_dim] + hidden_size + [1], activation=nn.ReLU()
        )

    def forward(self, t: int) -> None:
        """Compute Q-value estimate for state-action pair."""
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))

        state_action = torch.cat([obs, action], dim=1)
        q_values = self.model(state_action).squeeze(-1)
        advantages = -torch.abs(q_values)

        self.set((f"{self.prefix}q_value", t), q_values)
        self.set((f"{self.prefix}advantages", t), advantages)


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
        """Initialize gaussian policy network.

        * state_dim: Dimension of the state space
        * hidden_size: List of hidden layer sizes
        * action_dim: Dimension of the action space
        * log_std_min: Minimum value for log standard deviation
        * log_std_max: Maximum value for log standard deviation
        """
        super().__init__()

        # This policy is used in both SAC and AFU-alpha, implementing a
        # state-dependent Gaussian policy with tanh transformation to bound
        # actions. The network outputs both mean and log standard deviation of
        # the Gaussian distribution, allowing for learned state-independent
        # standard deviations.
        self.model = build_mlp(
            [state_dim] + hidden_size + [3 * action_dim],
            activation=nn.ReLU(),
        )
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, t: int) -> None:
        """Compute mean and log_std of the action distribution."""
        obs = self.get(("env/env_obs", t))
        output = self.model(obs)
        mean, log_std, extra_loc = torch.chunk(output, 3, dim=1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        extra_loc = torch.clamp(extra_loc, -1.0, 1.0)

        self.set(("mean", t), mean)
        self.set(("log_std", t), log_std)
        self.set(("extra_loc", t), extra_loc)

    def sample_action(self, workspace: Workspace, t: int) -> None:
        """Sample actions using the reparameterization trick."""
        self.workspace = workspace

        mean = self.get(("mean", t))
        log_std = self.get(("log_std", t))
        std = log_std.exp()

        # Implements the reparameterization trick for gradient propagation
        # through the sampling process, followed by tanh transformation
        # to bound actions.
        normal = torch.randn_like(mean)
        sample = mean + std * normal
        action = torch.tanh(sample)
        
        self.set(("sample", t), sample)
        self.set(("action", t), action)

    def get_log_prob(self, workspace: Workspace, t: int) -> None:
        """Compute log probability of an action under the Gaussian distribution."""
        self.workspace = workspace

        # Accounts for the tanh transformation when computing log probabilities
        # using the change of variables formula.
        mean = self.get(("mean", t))
        log_std = self.get(("log_std", t))
        sample = self.get(("sample", t))
        action = self.get(("action", t))
        
        std = log_std.exp()

        # Gaussian log probability
        normal_log_prob = (-0.5 * ((sample - mean) / std).pow(2) - log_std).sum(
            dim=-1
        )

        # Apply change of variables formula for tanh transformation
        log_prob = normal_log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(
            dim=-1
        )

        self.set(("log_prob", t), log_prob)


class AFU:
    """Actor Free critic Update (AFU) implementation."""

    def __init__(self, params: dict) -> None:
        """Initialize AFU agent with given parameters.

        * params: dictionary containing
            - env_name: name of the gymnasium environment
            - hidden_size: list of hidden layer sizes for network
            - replay_size: size of replay buffer
            - batch_size: mini-batch size for updates
            - gamma: discount factor
            - tau: soft update coefficient
            - learning_rate: learning rate for all optimizers
            - gradient_reduction: gradient reduction factor
            - max_episodes: maximum number of episodes for training
            - max_steps: maximum steps per episode
        """
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

        self.v1 = ContinuousVFunction(
            state_dim, params["hidden_size"], prefix="v1/"
        )
        self.v2 = ContinuousVFunction(
            state_dim, params["hidden_size"], prefix="v2/"
        )

        self.target_v1 = ContinuousVFunction(
            state_dim, params["hidden_size"], prefix="target_v1/"
        )
        self.target_v2 = ContinuousVFunction(
            state_dim, params["hidden_size"], prefix="target_v2/"
        )

        self.q1 = ContiniousQFunction(
            state_dim, action_dim, params["hidden_size"], prefix="q1/"
        )
        self.q2 = ContiniousQFunction(
            state_dim, action_dim, params["hidden_size"], prefix="q2/"
        )

        self.policy = GaussianPolicy(
            state_dim,
            params["hidden_size"],
            action_dim,
            params["log_std_min"],
            params["log_std_max"],
        )

        self.target_v1.load_state_dict(self.v1.state_dict())
        self.target_v2.load_state_dict(self.v2.state_dict())

        self.replay_buffer = ReplayBuffer(params["replay_size"])

        self.log_alpha = torch.zeros(1, requires_grad=True)

        self.v1_optimizer = torch.optim.Adam(
            self.v1.parameters(), lr=params["learning_rate"]
        )
        self.v2_optimizer = torch.optim.Adam(
            self.v2.parameters(), lr=params["learning_rate"]
        )
        self.q1_optimizer = torch.optim.Adam(
            self.q1.parameters(), lr=params["learning_rate"]
        )
        self.q2_optimizer = torch.optim.Adam(
            self.q2.parameters(), lr=params["learning_rate"]
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=params["learning_rate"]
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=params["learning_rate"]
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
        """Select action for given state.

        * state: current state
        * evaluation: if True, use deterministic policy

        -> selected action
        """
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
        self, states: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss value for value and advantage functions using AFU's method.

        * states: batch of states
        * actions: batch of actions
        * targets: batch of targets

        -> value_loss
        """
        workspace = Workspace()

        workspace.set("env/env_obs", 0, states)
        self.v1(workspace, t=0)
        self.v2(workspace, t=0)
        v1_values = workspace.get("v1/value", 0)
        v2_values = workspace.get("v2/value", 0)
        optim_values = torch.min(v1_values, v2_values)

        workspace.set("action", 0, actions)
        self.q1(workspace, t=0)
        self.q2(workspace, t=0)
        q1_values = workspace.get("q1/q_value", 0)
        q2_values = workspace.get("q2/q_value", 0)

        optim_advantages = torch.stack(
            [
                q1_values - optim_values,
                q2_values - optim_values,
            ]
        )

        up_case = (targets <= optim_values).float().detach()
        no_mix_case = (
            (targets <= optim_values + optim_advantages).float().detach()
        )

        mix_gd_optim_values = (1 - no_mix_case) * (
            ((1 - self.params["gradient_reduction"]) * optim_values).detach()
            + self.params["gradient_reduction"] * optim_values
        ) + no_mix_case * optim_values

        critic_loss = (
            optim_advantages.pow(2)
            + up_case * 2 * optim_advantages * (mix_gd_optim_values - targets)
            + (mix_gd_optim_values - targets).pow(2)
        ).mean()

        return critic_loss

    def _compute_actor_loss(
        self,
        states: torch.Tensor,
        sampled_actions: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute policy loss including entropy term.

        * states: batch of states
        * sampled_actions: actions sampled from policy
        * log_probs: log probabilities of sampled actions

        -> tuple of (policy_loss, temperature_loss)
        """
        workspace = Workspace()

        workspace.set("env/env_obs", 0, states)
        workspace.set("action", 0, sampled_actions)
        self.q1(workspace, t=0)
        self.q2(workspace, t=0)
        q1_values = workspace.get("q1/q_value", 0)
        q2_values = workspace.get("q2/q_value", 0)

        min_q_values = torch.min(q1_values, q2_values)

        alpha = self.log_alpha.exp()
        policy_loss = (alpha * log_probs - min_q_values).mean()

        return policy_loss, log_probs

    def _adjust_temperature(
        self, log_probs: torch.Tensor, states: torch.Tensor
    ) -> torch.Tensor:
        target_entropy = -self.action_dim

        current_entropy = -log_probs

        alpha_error = current_entropy - target_entropy
        alpha_loss = -(self.log_alpha * alpha_error.detach()).mean()

        return alpha_loss

    def _compute_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute target values for critic updates.

        * rewards: batch of rewards
        * next_states: batch of next states
        * dones: batch of done flags

        -> target values for critic update
        """
        # AFU uses the minimum of the target value networks to compute targets:
        # target_q = r + γ * min(V1_target(s'), V2_target(s'))

        workspace = Workspace()

        workspace.set("env/env_obs", 0, next_states)
        self.target_v1(workspace, t=0)
        self.target_v2(workspace, t=0)
        next_v1 = workspace.get("target_v1/value", 0)
        next_v2 = workspace.get("target_v2/value", 0)

        target_v = torch.min(next_v1, next_v2)

        targets = (
            rewards + (1 - dones) * self.params["gamma"] * target_v
        ).detach()

        noise_scale = 0.1 * torch.abs(targets).mean().detach()
        noise = noise_scale * torch.randn_like(targets)
        targets = targets + noise

        return targets

    def update(self) -> tuple[float, float, float, float]:
        """Perform one update step of all networks.

        -> tuple of (policy_loss, critic_loss, value_loss, alpha_loss)
        """
        # Implements AFU's core mechanism where V and Q are updated together to
        # maintain their relationship - V functions track maximums of Q values
        # while Q functions use V targets for bootstrappig.

        if len(self.replay_buffer) < self.params["batch_size"]:
            return 0.0, 0.0, 0.0, 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.params["batch_size"], continuous=True
        )

        targets = self._compute_targets(rewards, next_states, dones)

        self.v1_optimizer.zero_grad()
        self.v2_optimizer.zero_grad()
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()

        critic_loss = self._compute_critic_loss(states, actions, targets)
        critic_loss.backward()

        self.v1_optimizer.step()
        self.v2_optimizer.step()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        workspace = Workspace()
        workspace.set("env/env_obs", 0, states)
        self.policy(workspace, t=0)
        self.policy.sample_action(workspace, t=0)
        self.policy.get_log_prob(workspace, t=0)

        sampled_actions = workspace.get("action", 0)
        log_probs = workspace.get("log_prob", 0)

        self.policy_optimizer.zero_grad()
        policy_loss, log_probs = self._compute_actor_loss(
            states, sampled_actions, log_probs
        )
        policy_loss.backward()
        self.policy_optimizer.step()

        # self.alpha_optimizer.zero_grad()
        alpha_loss = self._adjust_temperature(log_probs, states)
        # alpha_loss.backward()
        # self.alpha_optimizer.step()

        self._soft_update(self.v1, self.target_v1)
        self._soft_update(self.v2, self.target_v2)

        self.total_steps += 1

        return (
            policy_loss.item(),
            critic_loss.item(),
            critic_loss.item(),
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
        """Save model parameters and training state."""
        save_dict = {
            "v1_state": self.v1.state_dict(),
            "v2_state": self.v2.state_dict(),
            "target_v1_state": self.target_v1.state_dict(),
            "target_v2_state": self.target_v2.state_dict(),
            "q1_state": self.q1.state_dict(),
            "q2_state": self.q2.state_dict(),
            "policy_state": self.policy.state_dict(),
            "v1_optimizer_state": self.v1_optimizer.state_dict(),
            "v2_optimizer_state": self.v2_optimizer.state_dict(),
            "q1_optimizer_state": self.q1_optimizer.state_dict(),
            "q2_optimizer_state": self.q2_optimizer.state_dict(),
            "policy_optimizer_state": self.policy_optimizer.state_dict(),
            "alpha_optimizer_state": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha,
            "params": self.params,
            "total_steps": self.total_steps,
        }
        torch.save(save_dict, path)

    def load(self, path: str) -> None:
        """Load model parameters and training state."""
        save_dict = torch.load(path)

        self.v1.load_state_dict(save_dict["v1_state"])
        self.v2.load_state_dict(save_dict["v2_state"])
        self.target_v1.load_state_dict(save_dict["target_v1_state"])
        self.target_v2.load_state_dict(save_dict["target_v2_state"])
        self.q1.load_state_dict(save_dict["q1_state"])
        self.q2.load_state_dict(save_dict["q2_state"])
        self.policy.load_state_dict(save_dict["policy_state"])

        self.v1_optimizer.load_state_dict(save_dict["v1_optimizer_state"])
        self.v2_optimizer.load_state_dict(save_dict["v2_optimizer_state"])
        self.q1_optimizer.load_state_dict(save_dict["q1_optimizer_state"])
        self.q2_optimizer.load_state_dict(save_dict["q2_optimizer_state"])
        self.policy_optimizer.load_state_dict(save_dict["policy_optimizer_state"])
        self.alpha_optimizer.load_state_dict(save_dict["alpha_optimizer_state"])

        self.log_alpha = save_dict["log_alpha"]
        self.params = save_dict["params"]
        self.total_steps = save_dict["total_steps"]

    @classmethod
    def load_agent(cls, path: str) -> "AFU":
        """Create new agent instance from saved parameters."""
        save_dict = torch.load(path)
        agent = cls(save_dict["params"])
        agent.load(path)
        return agent
