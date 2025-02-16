from gymnasium.envs.mujoco.ant_v4 import AntEnv
from gymnasium.envs.registration import register
import numpy as np


class AntEnvStudy(AntEnv):
    def __init__(self):
        super().__init__()

    def set_state(
        self,
        # Position states
        z_pos,
        orientation_w,
        orientation_x,
        orientation_y,
        orientation_z,
        front_left_hip,
        front_left_ankle,
        front_right_hip,
        front_right_ankle,
        back_left_hip,
        back_left_ankle,
        back_right_hip,
        back_right_ankle,
        # Velocity states
        x_vel,
        y_vel,
        z_vel,
        angular_vel_x,
        angular_vel_y,
        angular_vel_z,
        front_left_hip_vel,
        front_left_ankle_vel,
        front_right_hip_vel,
        front_right_ankle_vel,
        back_left_hip_vel,
        back_left_ankle_vel,
        back_right_hip_vel,
        back_right_ankle_vel,
        # Optional x,y position
        x_pos=None,
        y_pos=None,
    ):
        """Set the state of the ant.

        Args:
            z_pos (float): z-coordinate of torso
            orientation_[w,x,y,z] (float): orientation quaternion of torso
            [front/back]_[left/right]_[hip/ankle] (float): joint angles
            [x,y,z]_vel (float): linear velocities
            angular_vel_[x,y,z] (float): angular velocities
            [front/back]_[left/right]_[hip/ankle]_vel (float): joint velocities
            x_pos (float, optional): x position of torso
            y_pos (float, optional): y position of torso
        """
        # Construct position state (qpos)
        if x_pos is not None and y_pos is not None:
            qpos = np.array(
                [
                    x_pos,
                    y_pos,
                    z_pos,  # Cartesian position
                    orientation_w,
                    orientation_x,
                    orientation_y,
                    orientation_z,  # Quaternion orientation
                    front_left_hip,
                    front_left_ankle,  # Front left leg
                    front_right_hip,
                    front_right_ankle,  # Front right leg
                    back_left_hip,
                    back_left_ankle,  # Back left leg
                    back_right_hip,
                    back_right_ankle,  # Back right leg
                ]
            )
        else:
            # Keep current x,y position
            qpos = np.array(
                [
                    self.data.qpos[0],
                    self.data.qpos[1],
                    z_pos,  # Keep current x,y
                    orientation_w,
                    orientation_x,
                    orientation_y,
                    orientation_z,
                    front_left_hip,
                    front_left_ankle,
                    front_right_hip,
                    front_right_ankle,
                    back_left_hip,
                    back_left_ankle,
                    back_right_hip,
                    back_right_ankle,
                ]
            )

        # Construct velocity state (qvel)
        qvel = np.array(
            [
                x_vel,
                y_vel,
                z_vel,  # Linear velocity
                angular_vel_x,
                angular_vel_y,
                angular_vel_z,  # Angular velocity
                front_left_hip_vel,
                front_left_ankle_vel,  # Front left leg
                front_right_hip_vel,
                front_right_ankle_vel,  # Front right leg
                back_left_hip_vel,
                back_left_ankle_vel,  # Back left leg
                back_right_hip_vel,
                back_right_ankle_vel,  # Back right leg
            ]
        )

        self.set_state(qpos, qvel)

    def get_observation_space(self):
        """Returns bounds for observation space"""
        if self._exclude_current_positions_from_observation:
            # 105-dimensional observation space without x,y positions
            return (
                np.array([-np.inf] * 105, dtype=np.float64),
                np.array([np.inf] * 105, dtype=np.float64),
            )
        else:
            # 107-dimensional observation space including x,y positions
            return (
                np.array([-np.inf] * 107, dtype=np.float64),
                np.array([np.inf] * 107, dtype=np.float64),
            )

    def get_action_space(self):
        """Returns bounds for action space"""
        return (
            np.array([-1.0] * 8, dtype=np.float32),  # 8 joint torques
            np.array([1.0] * 8, dtype=np.float32),
        )


register(
    id="AntStudy-v0",
    entry_point="afu.environments.ant:AntEnvStudy",
    max_episode_steps=1000,
)
