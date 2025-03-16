from gymnasium.envs.registration import register
import gymnasium as gym
import gymnasium_robotics
import d4rl


class AntMazeEnvStudy:
    def __init__(self, env_name="AntMaze_UMaze-v5"):
        """Initialize the AntMaze environment.

        Args:
            env_name: The name of the AntMaze environment to use.
        """
        # Register all gymnasium_robotics environments
        gym.register_envs(gymnasium_robotics)

        # Create the environment
        self.env = gym.make(env_name)

        self.dataset = d4rl.qlearning_dataset(self.env)

    def reset(self):
        """Reset the environment to start a new episode."""
        return self.env.reset()

    def step(self, action):
        """Take a step in the environment."""
        return self.env.step(action)

    def close(self):
        """Close the environment."""
        self.env.close()

    def get_observation_space(self):
        """Get the observation space bounds."""
        # For AntMaze, observation space is complex (dictionary)
        # We'll extract the main observation bounds for simplicity
        obs_space = self.env.observation_space["observation"]
        return (obs_space.low, obs_space.high)

    def get_action_space(self):
        """Get the action space bounds."""
        act_space = self.env.action_space
        return (act_space.low, act_space.high)

    def get_obs(self):
        """Get the current observation."""
        # In a real implementation, you would need to flatten the observation
        # or handle the dictionary structure
        return self.env.unwrapped._get_obs()

    def set_state(self, *state_values):
        """Set the state of the environment.

        Note: This is not fully implemented for AntMaze as it requires
        setting the internal MuJoCo state properly.
        """
        # This would need to be implemented properly to set the MuJoCo state
        print("Warning: set_state not fully implemented for AntMaze")
        pass

    def get_dataset(self):
        """Get the offline dataset for training."""
        return self.dataset

    def unwrapped(self):
        """Get the unwrapped environment."""
        return self


# Register the environment
register(
    id="AntMazeStudy-v0",
    entry_point="afu.environments.antmaze:AntMazeEnvStudy",
    max_episode_steps=1000,
)
