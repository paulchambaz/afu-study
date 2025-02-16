from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnasium.envs.registration import register
import numpy as np


class PendulumEnvStudy(PendulumEnv):
    def set_state(self, theta, theta_dot):
        self.state = (theta, theta_dot)

    def get_observation_space(self):
        return (
            np.array([-1.0, -1.0, -8.0]),
            np.array([1.0, 1.0, 8.0])
        )
    
    def get_action_space(self):
        return (
            np.array([-2.0]),
            np.array([2.0])
        )


register(
    id="PendulumStudy-v0",
    entry_point="afu.environments.pendulum:PendulumEnvStudy",
    max_episode_steps=200,
)
