from bbrl_gymnasium.envs.continuous_cartpole import ContinuousCartPoleEnv  # type: ignore
import numpy as np
from gymnasium.envs.registration import register


class ContinuousCartPoleEnvStudy(ContinuousCartPoleEnv):
    def set_state(
        self, cart_position, cart_velocity, pole_angle, pole_angular_velocity
    ):
        self.state = (
            cart_position,
            cart_velocity,
            pole_angle,
            pole_angular_velocity,
        )

    def get_observation_space(self):
        return (
            np.array([-4.8, -8.0, -0.418, -8.0]),
            np.array([4.8, 8.0, 0.418, 8.0]),
        )

    def get_action_space(self):
        return (
            np.array([-1.0]),
            np.array([1.0]),
        )


register(
    id="CartPoleContinuousStudy-v0",
    entry_point="afu.environments.cartpole:ContinuousCartPoleEnvStudy",
    max_episode_steps=500,
)
