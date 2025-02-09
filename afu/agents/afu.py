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
        pass

    def forward(self, t: int) -> None:
        """Computes value function estimate for the state."""
        pass


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
        pass

    def forward(self, t: int) -> None:
        """Computes mean and log_std of the action distribution."""
        pass

    def sample_action(self, workspace: Workspace, t: int) -> None:
        """Samples actions using the reparameterization trick."""
        pass


class ContinuousQFunction(Agent):
    """Q-function network that estimates state-action values Q(s,a)."""

    def __init__(
        self, state_dim: int, hidden_size: list[int], action_dim: int
    ) -> None:
        pass

    def forward(self, t: int) -> None:
        """Computes Q-value estimate for the state-action pair."""
        pass


class AFU:
    """Actor Free critic Update implementation."""

    def __init__(self, params: dict) -> None:
        pass

    def _soft_update(
        self, source_network: nn.Module, target_network: nn.Module
    ) -> None:
        """Performs soft update of target network parameters."""
        pass

    def select_action(
        self, state: np.ndarray, evaluation: bool = False
    ) -> np.ndarray:
        """Selects an action for the given state."""
        pass

    def _compute_critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the critic loss using gradient reduction mechanism."""
        pass

    def _compute_actor_loss(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the actor loss and entropy terms."""
        pass

    def _compute_value_loss(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the value function loss."""
        pass

    def update(self) -> tuple[float, float, float, float]:
        """Performs a single update step of all networks."""
        pass

    def train(self) -> dict:
        """Executes the complete training loop."""
        pass

    def save(self, path: str) -> None:
        """Saves model parameters to a file."""
        pass

    def load(self, path: str) -> None:
        """Loads model parameters from a file."""
        pass

    @classmethod
    def load_agent(cls, path: str) -> "AFU":
        """Creates a new agent instance from saved parameters."""
        pass
