from afu_rljax.algorithm import AFU # type: ignore
import numpy as np
import gymnasium as gym
import jax
import random


class AFUPerrin:
    def __init__(self, params: dict) -> None:
        env = gym.make(params["env_name"])
        self.algo = AFU(
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
        env.close()

    def select_action(
        self, state: np.ndarray, evaluation: bool = False
    ) -> np.ndarray:
        return self.algo.select_action(state)

    def update(self):
        self.algo.update()
