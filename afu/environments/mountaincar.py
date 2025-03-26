from gymnasium.envs.classic_control.continuous_mountain_car import (
    Continuous_MountainCarEnv,
)
from gymnasium.envs.registration import register
import numpy as np


class ContinuousMountainCarEnvStudy(Continuous_MountainCarEnv):
    def _set_state(self, car_position, car_velocity):
        self.state = np.array([car_position, car_velocity], dtype=np.float32)

    def get_obs(self):
        return np.array(self.state, dtype=np.float32)

    def get_observation_space(self):
        return (self.low_state, self.high_state)

    def get_action_space(self):
        return (
            np.array([self.min_action], dtype=np.float32),
            np.array([self.max_action], dtype=np.float32),
        )


register(
    id="MountainCarContinuousStudy-v0",
    entry_point="afu.environments.mountaincar:ContinuousMountainCarEnvStudy",
    max_episode_steps=500,
)
