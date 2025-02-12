from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnasium.envs.registration import register


class PendulumEnvStudy(PendulumEnv):
    def set_state(
        self, theta, theta_dot
    ):
        self.state = (
            theta,
            theta_dot
        )


register(
    id="PendulumStudy-v0",
    entry_point="afu.environments.pendulum:PendulumEnvStudy",
    max_episode_steps=200,
)
