from gymnasium.envs.box2d.bipedal_walker import BipedalWalker
from gymnasium.envs.registration import register
import numpy as np


class BipedalWalkerEnvStudy(BipedalWalker):
    def set_state(
        self,
        hull_angle: float,
        hull_angular_velocity: float,
        vel_x: float,
        vel_y: float,
        leg1_angle: float,
        leg1_angular_velocity: float,
        leg2_angle: float,
        leg2_angular_velocity: float,
        ground_contact0: float,
        ground_contact1: float,
        ground_contact2: float,
        ground_contact3: float,
        lidar: np.ndarray,
        lidar_contact0: float,
        lidar_contact1: float,
        lidar_contact2: float,
        lidar_contact3: float,
        lidar_contact4: float,
        lidar_contact5: float,
        lidar_contact6: float,
        lidar_contact7: float,
        lidar_contact8: float,
        lidar_contact9: float,
    ):
        self.hull.angle = hull_angle
        self.hull.angularVelocity = hull_angular_velocity
        self.hull.linearVelocity[0] = vel_x
        self.hull.linearVelocity[1] = vel_y
        self.legs[0].angle = leg1_angle
        self.legs[0].angularVelocity = leg1_angular_velocity
        self.legs[1].angle = leg2_angle
        self.legs[1].angularVelocity = leg2_angular_velocity
        self.legs[0].ground_contact = ground_contact0
        self.legs[1].ground_contact = ground_contact1
        self.legs[2].ground_contact = ground_contact2
        self.legs[3].ground_contact = ground_contact3
        self.lidar = lidar
        self.legs[0].lidar_contact = lidar_contact0
        self.legs[1].lidar_contact = lidar_contact1
        self.legs[2].lidar_contact = lidar_contact2
        self.legs[3].lidar_contact = lidar_contact3
        self.legs[4].lidar_contact = lidar_contact4
        self.legs[5].lidar_contact = lidar_contact5
        self.legs[6].lidar_contact = lidar_contact6
        self.legs[7].lidar_contact = lidar_contact7
        self.legs[8].lidar_contact = lidar_contact8
        self.legs[9].lidar_contact = lidar_contact9

    def get_obs(self):
        return np.array(
            [
                self.hull.angle,
                self.hull.angularVelocity,
                self.hull.linearVelocity[0],
                self.hull.linearVelocity[1],
                self.legs[0].angle,
                self.legs[0].angularVelocity,
                self.legs[1].angle,
                self.legs[1].angularVelocity,
                self.legs[0].ground_contact,
                self.legs[1].ground_contact,
                self.legs[2].ground_contact,
                self.legs[3].ground_contact,
                *self.lidar,
                self.legs[0].lidar_contact,
                self.legs[1].lidar_contact,
                self.legs[2].lidar_contact,
                self.legs[3].lidar_contact,
                self.legs[4].lidar_contact,
                self.legs[5].lidar_contact,
                self.legs[6].lidar_contact,
                self.legs[7].lidar_contact,
                self.legs[8].lidar_contact,
                self.legs[9].lidar_contact,
            ],
            dtype=np.float32,
        )

    def get_observation_space(self):
        """Returns bounds for observation space"""
        return (
            np.array(
                [-np.pi, -5.0, -5.0, -5.0, -np.pi, -5.0, -np.pi, -5.0, 0.0, -np.pi, -5.0, -np.pi, -5.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                dtype=np.float32,
            ),
            np.array(
                [np.pi, 5.0, 5.0, 5.0, np.pi, 5.0, np.pi, 5.0, 5.0, np.pi, 5.0, np.pi, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                dtype=np.float32,
            ),
        )

    def get_action_space(self):
        """Returns bounds for action space"""
        return (
            np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )

register(
    id="BipedalWalkerStudy-v0",
    entry_point="afu.environments.bipedalwalker:BipedalWalkerEnvStudy",
    max_episode_steps=1600,
)
