import torch
import gymnasium as gym
import torch.nn as nn
import random
import numpy as np
from bbrl.agents import Agent  # type: ignore
from bbrl_utils.nn import build_mlp  # type: ignore
from bbrl.workspace import Workspace  # type: ignore
from gymnasium.spaces import Discrete  # type: ignore
from .memory import ReplayBuffer
from tqdm import tqdm  # type: ignore


def DiscreteQNetwork(TimeAgent, SeedableAgent, SerializableAgent):
    """Q-Network that estimates"""

    def __init__(
        self, state_dim: int, hidden_layers: list[int], action_dim: int, name="critic"
    ):
        super().__init__(name=name)

        self.model = build_mlp(
            [state_dim] + hidden_layers + [action_dim], activation=nn.ReLU()
        )

    def forward(self, t: int) -> None:
        """Compute Q-value for a given state."""
        state = self.get(("env/env_obs", t))
