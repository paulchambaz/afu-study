from gymnasium.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from gymnasium.envs.registration import register

class ContinuousMountainCarEnvStudy(Continuous_MountainCarEnv):
    def set_state(self, car_position, car_velocity):
        self.state = (car_position, car_velocity)


register(
    id="MountainCarContinuousStudy-v0",
    entry_point="afu.environments.mountaincar:ContinuousMountainCarEnvStudy",
    max_episode_steps=500,
)
