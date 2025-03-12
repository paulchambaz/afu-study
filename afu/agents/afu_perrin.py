from afu_rljax.algorithm import AFU  # type: ignore
from omegaconf import OmegaConf
import numpy as np
import gymnasium as gym
import random
from .memory import ReplayBuffer
import torch


class AFUPerrin:
    def __init__(self, hyperparameters: OmegaConf) -> None:
        self.params = hyperparameters

        self.train_env = gym.make(self.params.env_name)
        self.algo = AFU(
            num_agent_steps=100_000,
            state_space=self.train_env.observation_space,
            action_space=self.train_env.action_space,
            seed=random.randint(0, 100_000),
            tau=self.params.tau,
            lr_actor=self.params.learning_rate,
            lr_critic=self.params.learning_rate,
            lr_alpha=self.params.learning_rate,
            units_actor=(self.params.hidden_size, self.params.hidden_size),
            units_critic=(self.params.hidden_size, self.params.hidden_size),
            gradient_reduction=self.params.gradient_reduction,
            buffer_size=self.params.replay_size,
            variant="alpha",
            alg="AFU",
        )
        self.replay_buffer = ReplayBuffer(self.params.replay_size)
        self.total_steps = 0
        self.episode_reward = 0.0

    def select_action(self, state: np.ndarray, evaluation: bool = False) -> np.ndarray:
        if evaluation:
            return self.algo.select_action(state)
        else:
            return self.algo.explore(state)

    def update(self):
        if len(self.replay_buffer) > 0:
            (
                state,
                action,
                reward,
                next_state,
                done,
            ) = self.replay_buffer.get_latest()

            self.algo.buffer.append(
                state=state,
                action=action,
                reward=float(reward),
                done=float(done),
                next_state=next_state,
            )

            if self.total_steps % self.params["batch_size"] == 0:
                for _ in self.params["batch_size"]:
                    self.algo.update()

    @classmethod
    def _get_params_defaults(cls) -> OmegaConf:
        return OmegaConf.create(
            {
                "hidden_size": 128,
                "gradient_reduction": 0.8,
                "learning_rate": 3e-4,
                "tau": 0.01,
                "replay_size": 100_000,
                "batch_size": 128,
            }
        )

    @classmethod
    def _get_hp_space(cls):
        return {
            "hidden_size": ("int", 32, 256, True),
            "gradient_reduction": ("float", 0.5, 1.0, False),
            "learning_rate": ("float", 1e-5, 1e-2, True),
            "tau": ("float", 1e-4, 1e-1, True),
            "replay_size": ("int", 10_000, 1_000_000, True),
            "batch_size": ("int", 32, 512, True),
        }

    def save(self, path: str) -> None:
        """Save model parameters and training state."""
        save_dict = {
            "algo_state": self.algo.state_dict(),
            "replay_buffer": self.replay_buffer,
            "params": self.params,
            "total_steps": self.total_steps,
            "episode_reward": self.episode_reward,
        }
        torch.save(save_dict, path)
