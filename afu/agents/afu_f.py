import torch
import gymnasium as gym
import torch.nn as nn
import numpy as np
from bbrl.agents import Agent  # type: ignore
from bbrl_utils.nn import build_mlp  # type: ignore
from bbrl.utils.replay_buffer import ReplayBuffer  # type: ignore
from bbrl.workspace import Workspace  # type: ignore
from tqdm import tqdm  # type: ignore


class ContinuousVFunction(Agent):
    """Value function network that estimates state values V(S)."""

    def __init__(self, state_dim: int, hidden_size: list[int], prefix: str) -> None:
        """Initialize value function network.

        * state_dim: dimension of the state space
        * hidden_size: list of hidden layer sizes
        """
        super().__init__()
        self.prefix = prefix

        self.model = build_mlp([state_dim] + hidden_size + [1], activation=nn.ReLU())

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

        mean = self.get(("mean", t))
        log_std = self.get(("log_std", t))
        sample = self.get(("sample", t))
        action = self.get(("action", t))

        std = log_std.exp()

        # Accounts for the tanh transformation when computing log probabilities
        # using the change of variables formula.
        normal_log_prob = (-0.5 * ((sample - mean) / std).pow(2) - log_std).sum(dim=-1)

        # Apply change of variables formula for tanh transformation
        log_prob = normal_log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

        self.set(("log_prob", t), log_prob)


class AFU:
    """Actor-Free critic Update (AFU) implementation."""

    def __init__(self, params: dict, mode: str = "alpha") -> None:
        """Initialize AFU agent with Alpha or Beta mode.

        * params: Dictionary containing hyperparameters
        * mode: "alpha" (default) or "beta" (for improved actor updates)
        """
        self.params = params
        self.mode = mode  # Choose between "alpha" and "beta"

        # Create environment
        self.train_env = gym.make(params["env_name"])
        state_dim = self.train_env.observation_space.shape[0]
        action_dim = self.train_env.action_space.shape[0]
        self.action_dim = action_dim

        # Value & Critic networks
        self.v1 = ContinuousVFunction(state_dim, params["hidden_size"], prefix="v1/")
        self.v2 = ContinuousVFunction(state_dim, params["hidden_size"], prefix="v2/")
        self.target_v1 = ContinuousVFunction(state_dim, params["hidden_size"], prefix="target_v1/")
        self.target_v2 = ContinuousVFunction(state_dim, params["hidden_size"], prefix="target_v2/")
        self.q1 = ContiniousQFunction(state_dim, action_dim, params["hidden_size"], prefix="q1/")
        self.q2 = ContiniousQFunction(state_dim, action_dim, params["hidden_size"], prefix="q2/")

        # Policy network
        self.policy = GaussianPolicy(
            state_dim,
            params["hidden_size"],
            action_dim,
            params["log_std_min"],
            params["log_std_max"],
        )

        # If in AFU-Beta mode, initialize μ_ζ (deterministic actor guidance)
        if self.mode == "beta":
            self.mu_zeta = build_mlp([state_dim] + params["hidden_size"] + [action_dim], activation=nn.ReLU())
            self.mu_zeta_optimizer = torch.optim.Adam(self.mu_zeta.parameters(), lr=params["learning_rate"])

        self.replay_buffer = ReplayBuffer(params["replay_size"])

        self.log_alpha = torch.zeros(1, requires_grad=True)

        # Optimizers
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=params["learning_rate"])
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=params["learning_rate"])
        self.v1_optimizer = torch.optim.Adam(self.v1.parameters(), lr=params["learning_rate"])
        self.v2_optimizer = torch.optim.Adam(self.v2.parameters(), lr=params["learning_rate"])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=params["learning_rate"])
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=params["learning_rate"])

        self.total_steps = 0


    def _soft_update(self, source_network: nn.Module, target_network: nn.Module) -> None:
        """Perform a soft update using state_dict blending for efficiency."""
        for name, param in target_network.named_parameters():
            param.data.copy_(
                (1 - self.params["tau"]) * param.data + self.params["tau"] * source_network.state_dict()[name].data
            )


    def select_action(self, state: np.ndarray, evaluation: bool = False) -> np.ndarray:
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


    def _compute_critic_loss(self, workspace: Workspace) -> torch.Tensor:
        """Compute loss for Q-functions (critic) using AFU's method with BBRL.

        * workspace: BBRL Workspace containing state-action pairs and targets
        -> critic loss
        """
        states = workspace.get("env/env_obs", 0)
        actions = workspace.get("action", 0)
        targets = workspace.get("target_q", 0)

        q1_values = self.q1.model(torch.cat([states, actions], dim=1)).squeeze(-1)
        q2_values = self.q2.model(torch.cat([states, actions], dim=1)).squeeze(-1)

        loss_q1 = ((q1_values - targets) ** 2).mean()
        loss_q2 = ((q2_values - targets) ** 2).mean()

        return loss_q1 + loss_q2


    def _compute_actor_loss(self, workspace: Workspace) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute policy loss including entropy term, with mode-dependent updates.

        * workspace: BBRL Workspace containing states and actions
        -> tuple of (policy_loss, temperature_loss)
        """
        states = workspace.get("env/env_obs", 0)
        sampled_actions = workspace.get("action", 0)
        log_probs = workspace.get("log_prob", 0)
        q_values = self.q1.model(torch.cat([states, sampled_actions], dim=1)).squeeze(-1)

        # Standard SAC-style actor loss (used in both Alpha and Beta)
        policy_loss = (self.log_alpha.exp() * log_probs - q_values).mean()

        # Temperature loss (same for both)
        target_entropy = -self.action_dim
        temperature_loss = -(self.log_alpha.exp() * (log_probs + target_entropy)).mean()

        if self.mode == "beta":
            # Train µ_ζ(s) using regression
            mu_pred = self.mu_zeta(states)
            loss_mu = ((mu_pred - sampled_actions) ** 2).mean()

            self.mu_zeta_optimizer.zero_grad()
            loss_mu.backward()
            self.mu_zeta_optimizer.step()

            # Modify policy gradient to avoid local optima
            grad = torch.autograd.grad(policy_loss, sampled_actions, retain_graph=True)[0]
            correction = mu_pred - sampled_actions
            dot_product = (grad * correction).sum(dim=-1, keepdim=True)

            if (dot_product < 0).any():  # If gradients point in the wrong direction
                grad = grad - ((grad * correction).sum() / (correction.norm() ** 2 + 1e-6)) * correction

            policy_loss = (grad.detach() * sampled_actions).sum()

        return policy_loss, temperature_loss


    def _adjust_temperature(self, workspace: Workspace) -> torch.Tensor:
        """Adjust entropy temperature dynamically using BBRL.

        * workspace: Workspace with policy outputs
        -> entropy temperature loss
        """
        log_probs = workspace.get("log_prob", 0)

        target_entropy = -self.action_dim
        temperature_loss = -(self.log_alpha.exp() * (log_probs + target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        temperature_loss.backward()
        self.alpha_optimizer.step()

        return temperature_loss.item()


    def _compute_targets(self, workspace: Workspace) -> None:
        """Compute target values for critic updates using the target value networks."""
        
        rewards = workspace.get_full("reward")
        next_states = workspace.get_full("next_env/env_obs")
        dones = workspace.get_full("done")

        next_workspace = Workspace()
        next_workspace.set("env/env_obs", 0, next_states)

        # Compute target values using target V-networks
        self.target_v1(next_workspace, t=0)
        self.target_v2(next_workspace, t=0)

        next_v1 = next_workspace.get("target_v1/value", 0)
        next_v2 = next_workspace.get("target_v2/value", 0)

        # Take the minimum target value (stabilizes training)
        target_v = torch.min(next_v1, next_v2)

        # Compute Bellman target: target_q = r + γ * min(V1_target(s'), V2_target(s'))
        targets = (rewards + (1 - dones) * self.params["gamma"] * target_v).detach()

        # Add noise to targets for stability
        noise_scale = 0.1 * torch.abs(targets).mean().detach()
        noise = noise_scale * torch.randn_like(targets)
        targets = targets + noise

        # Store targets in the workspace
        workspace.set_full("target_q", targets)


    def update(self) -> tuple[float, float, float, float]:
        """Perform one update step for Alpha or Beta mode."""
        if self.replay_buffer.size() < self.params["batch_size"]:
            return 0, 0, 0, 0  # Not enough samples

        workspace = self.replay_buffer.get_shuffled(self.params["batch_size"])

        states = workspace.get_full("env/env_obs")
        actions = workspace.get_full("action")
        rewards = workspace.get_full("reward")
        next_states = workspace.get_full("next_env/env_obs")
        dones = workspace.get_full("done").float() 

        # Compute targets for critic loss
        self._compute_targets(workspace)

        # Compute critic loss
        critic_loss = self._compute_critic_loss(workspace)
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        critic_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # Compute value loss
        with torch.no_grad():
            q1_values = self.q1.model(torch.cat([states, 
                                                actions], dim=1)).squeeze(-1)
            q2_values = self.q2.model(torch.cat([states, 
                                                actions], dim=1)).squeeze(-1)
            min_q_values = torch.min(q1_values, q2_values)

        v1_values = self.v1.model(states).squeeze(-1)
        v2_values = self.v2.model(states).squeeze(-1)

        value_loss = ((v1_values - min_q_values) ** 2).mean() + ((v2_values - min_q_values) ** 2).mean()
        self.v1_optimizer.zero_grad()
        self.v2_optimizer.zero_grad()
        value_loss.backward()
        self.v1_optimizer.step()
        self.v2_optimizer.step()

        # Compute policy loss based on mode
        self.policy(workspace, t=0)
        self.policy.sample_action(workspace, t=0)
        policy_loss, temperature_loss = self._compute_actor_loss(workspace)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Adjust entropy temperature
        self.alpha_optimizer.zero_grad()
        temperature_loss.backward()
        self.alpha_optimizer.step()

        # Soft update target networks
        self._soft_update(self.v1, self.target_v1)
        self._soft_update(self.v2, self.target_v2)

        return policy_loss.item(), critic_loss.item(), value_loss.item(), temperature_loss.item()


    def train(self) -> dict:
        """Train the agent over multiple episodes using replay buffer."""
        episode_rewards = []
        progress = tqdm(range(self.params["max_episodes"]), desc="Training", leave=True)

        for episode in progress:
            state, _ = self.train_env.reset()
            episode_reward = 0.0

            for step in range(self.params["max_steps"]):
                action = self.select_action(state)

                next_state, reward, terminated, truncated, _ = self.train_env.step(action)
                done = terminated or truncated

                # Ensure proper batch dimensions
                workspace = Workspace()
                workspace.set_full("env/env_obs", torch.tensor([state], dtype=torch.float32))  
                workspace.set_full("action", torch.tensor([action], dtype=torch.float32))  
                workspace.set_full("reward", torch.tensor([[reward]], dtype=torch.float32))
                workspace.set_full("next_env/env_obs", torch.tensor([next_state], dtype=torch.float32))  
                workspace.set_full("done", torch.tensor([[done]], dtype=torch.float32))  

                self.replay_buffer.put(workspace)

                _, _, _, _ = self.update()

                state = next_state
                episode_reward += reward

                if done:
                    break

            episode_rewards.append(episode_reward)

            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                progress.set_postfix({"avg_reward": f"{avg_reward:.2f}"}, refresh=True)

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