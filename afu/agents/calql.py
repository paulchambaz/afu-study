import torch
from omegaconf import OmegaConf
import gymnasium as gym
import torch.nn as nn
import numpy as np
from bbrl.agents import Agent
from bbrl_utils.nn import build_mlp
from bbrl.workspace import Workspace
from .memory import ReplayBuffer
import math
import pickle


class QNetwork(Agent):
    """Neural network that estimates Q-value for state action pairs."""

    def __init__(
        self, state_dim: int, hidden_dims: list[int], action_dim: int, prefix: str
    ) -> None:
        super().__init__()
        self.prefix = prefix

        self.model = build_mlp(
            [state_dim + action_dim] + hidden_dims + [1], activation=nn.ReLU()
        )

    def forward(self, t: int) -> None:
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        state_action = torch.cat([obs, action], dim=1)

        q_value = self.model(state_action).squeeze(-1)
        self.set((f"{self.prefix}/q_value", t), q_value)


class PolicyNetwork(Agent):
    """Neural network that outputs a Gaussian distribution over actions."""

    def __init__(self, state_dim: int, hidden_dims: list[int], action_dim: int) -> None:
        super().__init__()

        self.model = build_mlp(
            [state_dim] + hidden_dims + [action_dim], activation=nn.ReLU()
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, t: int) -> None:
        """Compute mean and log_std of action distribution for a given state."""
        obs = self.get(("env/env_obs", t))

        mean = self.model(obs)
        log_std = self.log_std.expand_as(mean)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        self.set(("mean", t), mean)
        self.set(("log_std", t), log_std)

    def sample_action(self, workspace: Workspace, t: int) -> None:
        """Sample action from the gaussian distribution using reparameterization trick."""
        self.workspace = workspace

        mean = self.get(("mean", t))
        log_std = self.get(("log_std", t))
        std = log_std.exp()

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

        normal_log_prob = (
            -0.5 * ((sample - mean) / log_std.exp()).pow(2)
            - log_std
            - 0.5 * math.log(2 * math.pi)
        ).sum(dim=-1)

        log_prob = normal_log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        self.set(("log_prob", t), log_prob)


class CALQL:
    def __init__(self, hyperparameters: OmegaConf):
        self.params = hyperparameters
        self.train_env = gym.make(self.params.env_name)

        self.state_dim = self.train_env.observation_space.shape[0]
        self.action_dim = self.train_env.action_space.shape[0]

        hidden_dims = [self.params.hidden_size, self.params.hidden_size]

        self.q_network = QNetwork(
            state_dim=self.state_dim,
            hidden_dims=hidden_dims,
            action_dim=self.action_dim,
            prefix="q",
        )

        self.q_target_network = QNetwork(
            state_dim=self.state_dim,
            hidden_dims=hidden_dims,
            action_dim=self.action_dim,
            prefix="q_target",
        )

        for target_param, param in zip(
            self.q_target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(param.data)

        self.policy_network = PolicyNetwork(
            state_dim=self.state_dim,
            hidden_dims=hidden_dims,
            action_dim=self.action_dim,
        )

        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))

        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.params.q_lr
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
        self.alpha = self.params.alpha
        self.n = self.params.n
        self.total_steps = 0

        if "dataset_files" in self.params:
            self._initialize_mc_returns(self.params.dataset_files)
        else:
            self.mc_returns = {}

    def _initialize_mc_returns(self, dataset_files):
        dataset = []
        for file in dataset_files:
            with open(file, "rb") as f:
                results = pickle.load(f)
                dataset.extend(results)

        self.mc_returns = {}

        episode_ends = [i for i, (_, _, _, _, done) in enumerate(dataset) if done]
        episode_starts = [-1] + episode_ends[:-1]

        for start, end in zip(episode_starts, episode_ends):
            episode = dataset[start + 1 : end + 1]
            discounted_return = 0

            for state, _, reward, _, _ in reversed(episode):
                discounted_return = reward + self.gamma * discounted_return

                state_rounded = tuple(round(float(s), 4) for s in state.tolist())

                if state_rounded in self.mc_returns:
                    old_count = self.mc_returns.get(state_rounded + ("count",), 1)
                    old_value = self.mc_returns[state_rounded]
                    new_value = (old_value * old_count + discounted_return) / (
                        old_count + 1
                    )
                    self.mc_returns[state_rounded] = round(new_value, 4)
                    self.mc_returns[state_rounded + ("count",)] = old_count + 1
                else:
                    self.mc_returns[state_rounded] = round(discounted_return, 4)
                    self.mc_returns[state_rounded + ("count",)] = 1

        count_keys = [k for k in self.mc_returns.keys() if k[-1] == "count"]
        for k in count_keys:
            del self.mc_returns[k]

        self.min_mc_returns = self.mc_returns[
            min(self.mc_returns, key=self.mc_returns.get)
        ]

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
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size,
            continuous=True,
        )

        q_loss = self._compute_q_loss(states, actions, rewards, next_states, dones)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        self._soft_update(self.q_network, self.q_target_network)

        policy_loss = self._compute_policy_loss(states)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha_loss = self._compute_alpha_loss(states)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

    def _compute_q_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Q-network loss based on the formula."""
        return self._compute_td_loss(
            states, actions, rewards, next_states, dones
        ) + self.alpha * self._compute_con_loss(states, actions)

    def _compute_td_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the TD Q-network loss based on the formula.

        L_Q^TD (theta) = mean [
            (Q_theta (s, a) - r - gamma Q_(theta^target) (s', a_(s')))^2
        ]
        """
        prefix = "q"
        target_prefix = "q_target"
        q_network = self.q_network
        q_target_network = self.q_target_network

        q_workspace = Workspace()
        q_workspace.set("env/env_obs", 0, states)
        q_workspace.set("action", 0, actions)
        q_network(q_workspace, t=0)
        q_values = q_workspace.get(f"{prefix}/q_value", 0)

        policy_workspace = Workspace()
        policy_workspace.set("env/env_obs", 0, next_states)
        self.policy_network(policy_workspace, t=0)
        self.policy_network.sample_action(policy_workspace, t=0)
        next_actions = policy_workspace.get("action", 0)

        q_target_workspace = Workspace()
        q_target_workspace.set("env/env_obs", 0, next_states)
        q_target_workspace.set("action", 0, next_actions)
        q_target_network(q_target_workspace, t=0)
        q_targets = q_target_workspace.get(f"{target_prefix}/q_value", 0)

        q_targets = rewards + (1.0 - dones) * self.gamma * q_targets

        q_loss = 0.5 * torch.mean((q_values - q_targets.detach()) ** 2)

        return q_loss

    def _compute_con_loss(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute the CON Q-network loss based on the formula.

        L_Q^CON (theta) = mean [
          log(sum_i exp(Q_theta(s, a_R^i) - log(0.5^|A|))
              + sum_i exp(max(V_mu(s), Q_theta(s, a_s^i) - log pi(a_s^i|s))))
          - Q_theta(s, a)
        ]
        """
        prefix = "q"
        q_network = self.q_network

        # Part 1: Generate random actions and compute their Q-values : Q_theta (s, a^i_r)
        random_actions = (
            torch.rand((self.n, self.batch_size, self.action_dim), device=states.device)
            * 2
            - 1
        )

        # Density of uniform distribution: (1/2)^|A|
        # Log density: log((1/2)^|A|) = |A| * log(1/2) = -|A| * log(2)
        random_density = -self.action_dim * np.log(2)

        q_rand_terms = []
        for i in range(self.n):
            q_rand_workspace = Workspace()
            q_rand_workspace.set("env/env_obs", 0, states)
            q_rand_workspace.set("action", 0, random_actions[i])
            q_network(q_rand_workspace, t=0)
            q_rand = q_rand_workspace.get(f"{prefix}/q_value", 0)
            # Q(s, a_R^i) - log(0.5^|A|)
            q_rand_terms.append(q_rand - random_density)

        # Part 2: Generate actions from policy and compute their Q-values
        q_pi_terms = []
        for i in range(self.n):
            # Sample actions from policy and get log probabilities
            pi_workspace = Workspace()
            pi_workspace.set("env/env_obs", 0, states)
            self.policy_network(pi_workspace, t=0)
            self.policy_network.sample_action(pi_workspace, t=0)
            self.policy_network.get_log_prob(pi_workspace, t=0)
            pi_actions = pi_workspace.get("action", 0)
            log_pis = pi_workspace.get("log_prob", 0)

            # Compute Q-values for these policy actions
            q_pi_workspace = Workspace()
            q_pi_workspace.set("env/env_obs", 0, states)
            q_pi_workspace.set("action", 0, pi_actions)
            q_network(q_pi_workspace, t=0)
            q_pi = q_pi_workspace.get(f"{prefix}/q_value", 0)

            # Get reference values V_mu(s)
            v_mu = self._compute_reference_values(states)

            # Cal-QL modification: Apply max between Q-values and reference values
            # max(V_mu(s), Q(s, a_s^i) - log pi(a_s^i|s))
            q_pi_cal = torch.maximum(q_pi - log_pis, v_mu)

            q_pi_terms.append(q_pi_cal)

        # Part 3: Get Q-values for actions from the batch
        q_batch_workspace = Workspace()
        q_batch_workspace.set("env/env_obs", 0, states)
        q_batch_workspace.set("action", 0, actions)
        q_network(q_batch_workspace, t=0)
        q_batch = q_batch_workspace.get(f"{prefix}/q_value", 0)

        # Part 4: Combine terms and compute logsumexp
        # Stack all terms: [q_rand_terms + q_pi_terms]
        all_terms = torch.stack(q_rand_terms + q_pi_terms, dim=0)

        # Compute logsumexp: log(sum_i exp(Q_rand_terms) + sum_i exp(Q_pi_terms))
        logsumexp_term = torch.logsumexp(all_terms, dim=0)

        # Final loss: logsumexp_term - Q_batch
        con_loss = (logsumexp_term - q_batch).mean()

        return con_loss

    def _compute_reference_values(self, states: torch.Tensor) -> torch.Tensor:
        values = torch.zeros(self.batch_size, device=states.device)

        states_np = states.detach().numpy()

        for i in range(self.batch_size):
            state = tuple(round(float(s), 4) for s in states_np[i].tolist())

            if state in self.mc_returns:
                values[i] = self.mc_returns[state]
            else:
                values[i] = self._find_nearest_value(state)

        return values

    def _find_nearest_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.min_mc_returns

    def _compute_policy_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute the policy loss based on the formula.

        L_policy (phi.alt) = mean [ alpha * log(policy_(phi.alt) (a_s | s)) - Q_(theta_i) (s, a_s) ]
        """
        policy_workspace = Workspace()
        policy_workspace.set("env/env_obs", 0, states)
        self.policy_network(policy_workspace, t=0)
        self.policy_network.sample_action(policy_workspace, t=0)
        self.policy_network.get_log_prob(policy_workspace, t=0)
        actions = policy_workspace.get("action", 0)
        log_probs = policy_workspace.get("log_prob", 0)

        q_workspace = Workspace()
        q_workspace.set("env/env_obs", 0, states)
        q_workspace.set("action", 0, actions)
        self.q_network(q_workspace, t=0)
        q_values = q_workspace.get("q/q_value", 0)

        alpha = self.log_alpha.exp()
        policy_loss = (alpha * log_probs - q_values).mean()

        return policy_loss

    def _compute_alpha_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute the temperature parameter loss.

        L_temp (alpha) = [ -log (policy_theta (a_s | s)) - alpha target_entropy ]
        """
        policy_workspace = Workspace()
        policy_workspace.set("env/env_obs", 0, states)
        self.policy_network(policy_workspace, t=0)

        self.policy_network.sample_action(policy_workspace, t=0)
        self.policy_network.get_log_prob(policy_workspace, t=0)
        log_probs = policy_workspace.get("log_prob", 0)

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
            "q_target_network_state": self.q_target_network.state_dict(),
            "policy_network_state": self.policy_network.state_dict(),
            # Optimizer states
            "q_optimizer_state": self.q_optimizer.state_dict(),
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
        self.q_network.load_state_dict(save_dict["q_network_state"])
        self.q_target_network.load_state_dict(save_dict["q_target_network_state"])
        self.policy_network.load_state_dict(save_dict["policy_network_state"])

        # Restore optimizer states
        self.q_optimizer.load_state_dict(save_dict["q_optimizer_state"])
        self.policy_optimizer.load_state_dict(save_dict["policy_optimizer_state"])
        self.alpha_optimizer.load_state_dict(save_dict["alpha_optimizer_state"])

        # Restore temperature parameter
        with torch.no_grad():
            self.log_alpha.copy_(save_dict["log_alpha"])

        # Restore other parameters
        self.params = save_dict["params"]

    @classmethod
    def loadagent(cls, path: str) -> "CALQL":
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
                "alpha": 5,
                "n": 4,
            }
        )
