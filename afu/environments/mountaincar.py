import gymnasium as gym

class MountainCarEnv:
    """
    Simple wrapper around MountainCarContinuous environment that provides basic
    functionality for training and visualization.
    """
    def __init__(self, render_mode=None):
        self.env = gym.make("MountainCarContinuous-v0", render_mode=render_mode)
        self.state_dim = 2  # position and velocity
        self.action_dim = 1  # force applied to car
        
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
        
    def close(self):
        self.env.close()
        
    def run_episodes(self, num_episodes=10, max_steps=999, policy=None):
        """
        Run multiple episodes, either with a provided policy or random actions.
        Records total reward and steps for each episode.
        """
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            observation, _ = self.env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = (
                    policy(observation)
                    if policy
                    else self.env.action_space.sample()
                )
                observation, reward, done, truncated, _ = self.env.step(action)
                total_reward += reward
                
                if done or truncated:
                    break
                    
            episode_rewards.append(total_reward)
            episode_lengths.append(step + 1)
            print(
                f"Episode {episode + 1}: Steps = {step + 1}, Reward = {total_reward}"
            )
            
        return episode_rewards, episode_lengths

    def is_continuous_action(self):
        return True

    def get_obs_and_actions_sizes(self):
        return self.state_dim, self.action_dim
