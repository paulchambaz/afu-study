from gymnasium.envs.mujoco.swimmer_v4 import SwimmerEnv
from gymnasium.envs.registration import register
import numpy as np

class SwimmerEnvStudy(SwimmerEnv):
    def __init__(self):
        super().__init__()
    
    def set_state(
        self, 
        front_angle,
        rot1_angle, 
        rot2_angle,
        tip_vel_x,
        tip_vel_y,
        front_angular_vel,
        rot1_angular_vel,
        rot2_angular_vel,
        x_pos=None,
        y_pos=None
    ):
        """Set the state of the swimmer.
        
        Args:
            front_angle (float): angle of the front tip
            rot1_angle (float): angle of first rotor
            rot2_angle (float): angle of second rotor
            tip_vel_x (float): velocity of tip along x-axis
            tip_vel_y (float): velocity of tip along y-axis 
            front_angular_vel (float): angular velocity of front tip
            rot1_angular_vel (float): angular velocity of first rotor
            rot2_angular_vel (float): angular velocity of second rotor
            x_pos (float, optional): x position of tip
            y_pos (float, optional): y position of tip
        """
        if x_pos is not None and y_pos is not None:
            qpos = np.array([x_pos, y_pos, front_angle, rot1_angle, rot2_angle])
        else:
            qpos = np.array([
                self.data.qpos[0],
                self.data.qpos[1],
                front_angle,
                rot1_angle,
                rot2_angle
            ])
            
        qvel = np.array([
            tip_vel_x,
            tip_vel_y,
            front_angular_vel,
            rot1_angular_vel,
            rot2_angular_vel
        ])
        
        self.set_state(qpos, qvel)
        
    def get_observation_space(self):
        """Returns bounds for observation space"""
        if self._exclude_current_positions_from_observation:
            return (
                np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float64),
                np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
            )
        else:
            return (
                np.array([-np.inf]*10, dtype=np.float64),
                np.array([np.inf]*10, dtype=np.float64)
            )
    
    def get_action_space(self):
        """Returns bounds for action space"""
        return (
            np.array([-1.0, -1.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32)
        )

register(
    id="SwimmerStudy-v0",
    entry_point="afu.environments.swimmer:SwimmerEnvStudy",
    max_episode_steps=1000,
)
