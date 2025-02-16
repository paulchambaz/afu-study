from gymnasium.envs.box2d.bipedal_walker import BipedalWalker
from gymnasium.envs.registration import register


class BipedalWalkerEnvStudy(BipedalWalker):
    def set_state(
        self,
        hull_angle,
        hull_angular_velocity,
        leg1_angle,
        leg1_speed,
        leg2_angle,
        leg2_speed,
    ):
        self.hull.angle = hull_angle
        self.hull.angularVelocity = hull_angular_velocity
        self.legs[0].angle = leg1_angle
        self.legs[0].speed = leg1_speed
        self.legs[1].angle = leg2_angle
        self.legs[1].speed = leg2_speed


register(
    id="BipedalWalkerStudy-v0",
    entry_point="afu.environments.bipedalwalker:BipedalWalkerEnvStudy",
    max_episode_steps=1600,
)
