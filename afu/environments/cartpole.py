from bbrl_gymnasium.envs.continuous_cartpole import ContinuousCartPoleEnv # type: ignore
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

register(
    id="CartPoleContinuousStudy-v0",
    entry_point="afu.environments.cartpole:ContinuousCartPoleEnvStudy",
    max_episode_steps=500,
)
