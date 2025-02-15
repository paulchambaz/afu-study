from .base import Experiment
import numpy as np
from tqdm import tqdm  # type: ignore


class OnPolicy(Experiment):
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
            "max_episodes": self.params["total_episodes"],
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

        for i in range(self.param["n"]):
            training_steps = 0

            episode_rewards = []
            agent = self.algo(params)
            progress = tqdm(
                range(params["max_episodes"]), desc=f"Training {i}/{self.params['n']}"
            )

            for episode in progress:
                state, _ = agent.train_env.reset()
                episode_reward = 0.0

                for step in range(agent.params["max_steps"]):
                    action = agent.select_action(state)
                    (
                        next_state,
                        reward,
                        terminated,
                        truncated,
                        _,
                    ) = agent.train_env.step(action)
                    done = terminated or truncated

                    agent.replay_buffer.push(state, action, reward, next_state, done)
                    agent.update()

                    state = next_state
                    episode_reward += float(reward)
                    agent.total_steps += 1
                    training_steps += 1

                    if training_steps % self.params["interval"] == 0:
                        results = self.evaluation(agent, self.env_name)
                        id = training_steps / self.params["interval"]
                        if id not in training_rewards:
                            training_rewards[id] = []
                        training_rewards[id].extend(results)

                    if done:
                        break

                episode_rewards.append(episode_reward)

                if len(episode_rewards) >= 10:
                    avg_reward = np.mean(episode_rewards[-10:])
                    progress.set_postfix(
                        {"avg_reward": f"{avg_reward:.2f}"}, refresh=True
                    )

        self.save_results()
