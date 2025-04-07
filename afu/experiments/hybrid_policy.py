from .base import Experiment
import numpy as np
from tqdm import tqdm  # type: ignore
import gymnasium as gym


class HybridPolicy(Experiment):
    def run(self, i, shared_results, results_lock, manager):
        # epsilons = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # epsilons = [0.0, 0.25, 0.5, 0.75, 1.0]
        epsilons = [0.4]
        obs_scale_factor = 1.0

        for epsilon in epsilons:
            training_steps = 0
            agent = self.algo(self.hyperparameters)

            on_policy_env = gym.make(self.params.env_name)
            off_policy_env = gym.make(self.params.env_name)

            on_policy_state, _ = on_policy_env.reset()

            progress = tqdm(
                range(self.params.total_steps),
                desc=f"Training {i}/{self.params.n} (epsilon={epsilon})",
            )

            while training_steps < self.params.total_steps:
                if np.random.random() < epsilon:
                    # off policy approach: random state and action
                    off_policy_state, _ = off_policy_env.reset()

                    obs_low, obs_high = (
                        agent.train_env.unwrapped.get_observation_space()
                    )
                    random_state = np.random.uniform(
                        low=obs_low * obs_scale_factor, high=obs_high * obs_scale_factor
                    )
                    off_policy_env.unwrapped._set_state(*random_state)
                    state = off_policy_env.unwrapped.get_obs()

                    act_low, act_high = agent.train_env.unwrapped.get_action_space()

                    action = np.random.uniform(
                        low=-1.0, high=1.0, size=self.action_space[0].shape
                    )
                    action = self._scale_action(action, self.action_space)

                    next_state, reward, terminated, truncated, _ = off_policy_env.step(
                        action
                    )
                    done = terminated or truncated

                    agent.replay_buffer.push(state, action, reward, next_state, done)
                else:
                    # on-policy approach: agent-selected action
                    state = on_policy_state

                    action = agent.select_action(state)
                    action = self._scale_action(action, self.action_space)

                    next_state, reward, terminated, truncated, _ = on_policy_env.step(
                        action
                    )
                    done = terminated or truncated

                    agent.replay_buffer.push(state, action, reward, next_state, done)

                    if done:
                        on_policy_state, _ = on_policy_env.reset()
                    else:
                        on_policy_state = next_state

                agent.update()

                agent.total_steps += 1
                training_steps += 1

                progress.update(1)

                if training_steps % self.params.interval == 0:
                    eval_results = self.evaluation(agent)
                    id = training_steps // self.params.interval

                    with results_lock:
                        if epsilon not in shared_results["rewards"]:
                            shared_results["rewards"][epsilon] = manager.dict()

                        if id not in shared_results["rewards"][epsilon]:
                            shared_results["rewards"][epsilon][id] = eval_results
                        else:
                            current_results = shared_results["rewards"][epsilon][id]
                            shared_results["rewards"][epsilon][id] = (
                                current_results + eval_results
                            )

                    progress.set_postfix({"eval": self._get_stats(eval_results)})

                if training_steps >= self.params.total_steps:
                    break

            on_policy_env.close()
            off_policy_env.close()

            with results_lock:
                if "agent" not in shared_results:
                    shared_results["agent"] = manager.dict()
                shared_results["agent"][epsilon] = agent.get_weights()
