from afu_rljax.algorithm import AFU  # type: ignore
from omegaconf import OmegaConf
import numpy as np
import gymnasium as gym
import random
from .memory import ReplayBuffer


class AFUPerrin:
    def __init__(self, **kwargs) -> None:
        self.params = OmegaConf.merge(
            AFUPerrin._get_params_defaults(),
            OmegaConf.create(kwargs),
        )

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
            units_actor=(self.params.hidden_size[0], self.params.hidden_size[1]),
            units_critic=(self.params.hidden_size[0], self.params.hidden_size[1]),
            gradient_reduction=self.params.gradient_reduction,
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
        if len(self.replay_buffer) >= self.params["batch_size"]:
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

            self.algo.update()

    @classmethod
    def _get_params_defaults(cls) -> OmegaConf:
        return OmegaConf.create(
            {
                "hidden_size": [128, 128],
                "gradient_reduction": 0.8,
                "learning_rate": 3e-4,
                "tau": 0.01,
                "replay_size": 100_000,
                "batch_size": 128,
            }
        )
