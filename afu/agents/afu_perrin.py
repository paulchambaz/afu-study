from afu_rljax.algorithm import AFU  # type: ignore
import numpy as np
import gymnasium as gym
import random
from .memory import ReplayBuffer


class AFUPerrin:
    def __init__(self, params: dict) -> None:
        env = gym.make(params["env_name"])
        self.params = params
        self.train_env = env
        self.algo = AFU(
            num_agent_steps=100_000,
            state_space=env.observation_space,
            action_space=env.action_space,
            seed=random.randint(0, 100_000),
            tau=params["tau"],
            lr_actor=params["learning_rate"],
            lr_critic=params["learning_rate"],
            lr_alpha=params["learning_rate"],
            units_actor=(params["hidden_size"][0], params["hidden_size"][1]),
            units_critic=(params["hidden_size"][0], params["hidden_size"][1]),
            gradient_reduction=params["gradient_reduction"],
            variant="alpha",
            alg="AFU",
        )

        self.replay_buffer = ReplayBuffer(params["replay_size"])
        self.total_steps = 0
        self.episode_reward = 0.0

    def select_action(
        self, state: np.ndarray, evaluation: bool = False
    ) -> np.ndarray:
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
