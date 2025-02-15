from .base import Experiment
import gymnasium as gym
import numpy as np
from tqdm import tqdm  # type: ignore


class OffPolicy(Experiment):
    def run(self):
        params = {
            "env_name": self.env_name,
            "actor_hidden_size": [128, 128],
            "critic_hidden_size": [128, 128],
            "noise_std": 0.1,
            "replay_size": 100_000,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "tau": 0.01,
            "gamma": 0.99,
            "batch_size": 128,
            "max_episodes": 500,
            "max_steps": 500,
            "policy_hidden_size": [128, 128],
            "q_hidden_size": [128, 128],
            "policy_lr": 3e-4,
            "q_lr": 3e-4,
            "alpha_lr": 3e-4,
            "hidden_size": [128, 128],
            "log_std_min": -20.0,
            "log_std_max": 2.0,
            "gradient_reduction": 0.8,
            "learning_rate": 3e-4,
        }

        training_rewards = {}

        env = gym.make(self.env_name)
        for i in range(self.params["n"]):
            training_steps = 0
            agent = self.algo(params)
            rewards = []

            progress = tqdm(
                range(self.params["total_steps"]),
                desc=f"Training {i}/{self.params['n']}",
            )

            for step in progress:
                agent.train_env.reset()

                obs_low, obs_high = self.observation_space
                random_state = np.random.uniform(low=obs_low * 0.5, high=obs_high * 0.5)
                agent.train_env.unwrapped.set_state(*random_state)

                act_low, act_high = self.action_space
                action = np.random.uniform(low=act_low, high=act_high)
                next_state, reward, terminated, truncated, _ = agent.train_env.step(
                    action
                )
                done = terminated or truncated

                agent.replay_buffer.push(random_state, action, reward, next_state, done)
                agent.update()

                reward += float(reward)
                agent.total_steps += 1
                training_steps += 1

                self.results["metadata"]["total_steps"] += 1
                rewards.append(reward)

                # just for display
                if len(rewards) >= 10:
                    metrics = {
                        "avg_reward": np.mean(rewards[-10:]),
                        "steps": self.results["metadata"]["total_steps"]
                    }
                    self.log_metrics(step, metrics)
                    progress.set_postfix({"avg_reward": f"{metrics['avg_reward']:.2f}"})

                if training_steps % self.params["interval"] == 0:
                    results = self.evaluation(agent, self.env_name)
                    id = training_steps // self.params["interval"]
                    if id not in training_rewards:
                        training_rewards[id] = []
                    training_rewards[id].extend(results)

        env.close()
        self.save_results()
