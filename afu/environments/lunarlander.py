from gymnasium.envs.box2d.lunar_lander import LunarLander
from gymnasium.envs.registration import register
import numpy as np


class LunarLanderContinuousStudy(LunarLander):
    def __init__(self, *args, **kwargs):
        render_mode = kwargs.pop("render_mode", None)
        super().__init__(*args, continuous=True)
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
        """Set the state of the lunar lander.

        Args:
            x_pos (float): x position [-1.5, 1.5]
            y_pos (float): y position [-1.5, 1.5]
            x_vel (float): x velocity [-5, 5]
            y_vel (float): y velocity [-5, 5]
            angle (float): angle in radians [-pi, pi]
            angular_vel (float): angular velocity [-5, 5]
            left_contact (bool): left leg ground contact (0 or 1)
            right_contact (bool): right leg ground contact (0 or 1)
        """
        # Convert state to internal coordinate system
        viewport_w = self.VIEWPORT_W / self.SCALE
        viewport_h = self.VIEWPORT_H / self.SCALE

        # Convert normalized positions back to world coordinates
        world_x = (x_pos * viewport_w / 2) + viewport_w / 2
        world_y = (y_pos * viewport_h / 2) + (
            self.helipad_y + self.LEG_DOWN / self.SCALE
        )

        # Set lander position and velocities
        self.lander.position = (world_x, world_y)
        self.lander.linearVelocity.x = x_vel * (viewport_w / 2) / self.FPS
        self.lander.linearVelocity.y = y_vel * (viewport_h / 2) / self.FPS
        self.lander.angle = angle
        self.lander.angularVelocity = angular_vel * self.FPS / 20.0

        # Set leg contacts
        self.legs[0].ground_contact = bool(left_contact)
        self.legs[1].ground_contact = bool(right_contact)

        # Wake the lander if it was sleeping
        self.lander.awake = True

    def get_observation_space(self):
        """Returns bounds for observation space"""
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
        """Returns bounds for action space"""
        return (
            np.array([-1.0, -1.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
        )


register(
    id="LunarLanderContinuousStudy-v0",
    entry_point="afu.environments.lunarlander:LunarLanderContinuousStudy",
    max_episode_steps=1000,
)
