from gymnasium.envs.box2d.lunar_lander import (
    LunarLander,
    VIEWPORT_W,
    VIEWPORT_H,
    SCALE,
    FPS,
    LEG_DOWN,
)
from gymnasium.envs.registration import register
import numpy as np


class LunarLanderContinuousStudy(LunarLander):
    def __init__(self, *args, **kwargs):
        render_mode = kwargs.pop("render_mode", None)
        super().__init__(*args, continuous=True, **kwargs)
        self.render_mode = render_mode

    def set_state(
        self,
        x_pos,
        y_pos,
        x_vel,
        y_vel,
        angle,
        angular_vel,
        left_contact,
        right_contact,
    ):
        # Convert state to internal coordinate system
        viewport_w = VIEWPORT_W / SCALE
        viewport_h = VIEWPORT_H / SCALE

        # Convert normalized positions back to world coordinates
        world_x = (x_pos * viewport_w / 2) + viewport_w / 2
        world_y = (y_pos * viewport_h / 2) + (self.helipad_y + LEG_DOWN / SCALE)

        # Set lander position and velocities
        self.lander.position = (world_x, world_y)
        self.lander.linearVelocity.x = x_vel * (viewport_w / 2) / FPS
        self.lander.linearVelocity.y = y_vel * (viewport_h / 2) / FPS
        self.lander.angle = angle
        self.lander.angularVelocity = angular_vel * FPS / 20.0

        # Set leg contacts
        self.legs[0].ground_contact = bool(left_contact)
        self.legs[1].ground_contact = bool(right_contact)

        # Wake the lander if it was sleeping
        self.lander.awake = True

    def get_obs(self):
        pos = self.lander.position
        vel = self.lander.linearVelocity

        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]

        return np.array(state, dtype=np.float32)

    def get_observation_space(self):
        return (
            np.array(
                [-2.5, -2.5, -10.0, -10.0, -2 * np.pi, -10.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            np.array(
                [2.5, 2.5, 10.0, 10.0, 2 * np.pi, 10.0, 1.0, 1.0], dtype=np.float32
            ),
        )

    def get_action_space(self):
        return (
            np.array([-1.0, -1.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
        )


register(
    id="LunarLanderContinuousStudy-v0",
    entry_point="afu.environments.lunarlander:LunarLanderContinuousStudy",
    max_episode_steps=1000,
)
