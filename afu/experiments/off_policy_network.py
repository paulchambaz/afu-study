from .base import Experiment
import numpy as np
from tqdm import tqdm  # type: ignore


class OffPolicyNetwork(Experiment):
    def run(self, i, shared_results, results_lock):
        networks = [32, 128, 512, 1024, 2048]
        obs_scale_factor = 1.0

        for network in networks:
            training_steps = 0
            self.hyperparameters["hidden_size"] = network
            agent = self.algo(self.hyperparameters)

            progress = tqdm(
                range(self.params.total_steps),
                desc=f"Training {i}/{self.params.n} (network={network})",
            )

            while training_steps < self.params.total_steps:
                state, _ = agent.train_env.reset()

                obs_low, obs_high = agent.train_env.unwrapped.get_observation_space()
                random_state = np.random.uniform(
                    low=obs_low * obs_scale_factor, high=obs_high * obs_scale_factor
                )
                agent.train_env.unwrapped._set_state(*random_state)
                state = agent.train_env.unwrapped.get_obs()

                act_low, act_high = agent.train_env.unwrapped.get_action_space()

                action = np.random.uniform(
                    low=-1.0, high=1.0, size=self.action_space[0].shape
                )
                action = self._scale_action(action, self.action_space)

                next_state, reward, terminated, truncated, _ = agent.train_env.step(
                    action
                )
                done = terminated or truncated

                agent.replay_buffer.push(state, action, reward, next_state, done)

                agent.update()

                agent.total_steps += 1
                training_steps += 1

                progress.update(1)

                if training_steps % self.params["interval"] == 0:
                    eval_results = self.evaluation(agent)
                    id = training_steps // self.params.interval

                    with results_lock:
                        if id not in shared_results["rewards"]:
                            shared_results["rewards"][network] = {}

                        if id not in shared_results["rewards"][network]:
                            shared_results["rewards"][network][id] = eval_results
                        else:
                            current_results = shared_results["rewards"][network][id]
                            shared_results["rewards"][reward][id] = (
                                current_results + eval_results
                            )

                    progress.set_postfix({"eval": self._get_stats(eval_results)})

                if training_steps >= self.params["total_steps"]:
                    break

            with results_lock:
                if "agent" not in shared_results:
                    shared_results["agent"] = {}

                shared_results["agent"][network] = agent.get_weights()
