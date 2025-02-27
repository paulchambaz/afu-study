from .base import Experiment
import numpy as np
from tqdm import tqdm  # type: ignore


class OffPolicy(Experiment):
    def run(self):
        obs_scale_factor = 0.5

        for i in range(self.params["n"]):
            training_steps = 0
            agent = self.algo(env_name=self.env_name)

            progress = tqdm(
                range(self.params["total_steps"]),
                desc=f"Training {i}/{self.params['n']}",
            )

            for step in progress:
                agent.train_env.reset()

                obs_low, obs_high = agent.train_env.unwrapped.get_observation_space()
                random_state = np.random.uniform(
                    low=obs_low * obs_scale_factor, high=obs_high * obs_scale_factor
                )
                agent.train_env.unwrapped.set_state(*random_state)
                state = agent.train_env.unwrapped.get_obs()

                act_low, act_high = agent.train_env.unwrapped.get_action_space()
                action = np.random.uniform(low=-1.0, high=1.0, size=(1,))

                next_state, reward, terminated, truncated, _ = agent.train_env.step(
                    action
                )
                done = terminated or truncated

                agent.replay_buffer.push(state, action, reward, next_state, done)

                agent.update()

                agent.total_steps += 1
                training_steps += 1

                self.results["metadata"]["total_steps"] += 1

                if training_steps % self.params["interval"] == 0:
                    results = self.evaluation(agent)

                    id = training_steps // self.params["interval"]
                    if id not in self.results["rewards"]:
                        self.results["rewards"][id] = []
                    self.results["rewards"][id].extend(results)

                    progress.set_postfix({"eval": self._get_stats(results)})

        self.save_results()
