from .base import Experiment
<<<<<<< HEAD
import numpy as np


class OfflineOnlineTransition(Experiment):
    def run(self, j, shared_results, results_lock, manager):
        agent = self.algo(self.hyperparameters)

        dataset = agent.train_env.unwrapped.dataset

        total_samples = len(dataset["observations"])
        remaining_indices = np.arange(total_samples)
        np.random.shuffle(remaining_indices)

        training_step = 0
        offline_step = 0

        for step in range(self.params.offline_steps):
            batch_size = min(self.params.batch_size, len(remaining_indices))

            batch_indices = remaining_indices[:batch_size]
            remaining_indices = remaining_indices[batch_size:]

            states = dataset["observations"][batch_indices]
            actions = dataset["actions"][batch_indices]
            rewards = dataset["rewards"][batch_indices]
            next_states = dataset["next_observations"][batch_indices]
            dones = dataset["terminals"][batch_indices]

            for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
                agent.replay_buffer.push(s, a, r, ns, d)

            agent.update()
            agent.total_steps += 1
            offline_step += 1
            training_step += 1

            if training_step % self.params.offline_interval == 0:
                results = self.evaluation(agent)
                id = training_step // self.params.offline_interval
                with results_lock:
                    if id not in shared_results["rewards"]:
                        shared_results["rewards"][id] = results
                    else:
                        current_results = shared_results["rewards"][id]
                        shared_results["rewards"][id] = current_results + results

                # Store the transition point for visualization
                with results_lock:
                    if "offline_transition" not in shared_results:
                        shared_results["offline_transition"] = training_step
=======
from tqdm import tqdm
import pickle


class Offline(Experiment):
    def run(self, i, shared_results, results_lock, manager):
        training_step = 0

        dataset_files = [
            "dataset/OnPolicy-AFU-PendulumStudy-v0-data.pk",
            "dataset/OffPolicy-AFU-PendulumStudy-v0-data.pk",
            "dataset/OnPolicy-SAC-PendulumStudy-v0-data.pk",
            "dataset/OffPolicy-SAC-PendulumStudy-v0-data.pk",
        ]

        self.hyperparameters["dataset_files"] = dataset_files

        dataset = []
        for file in dataset_files:
            with open(file, "rb") as f:
                results = pickle.load(f)
                dataset.extend(results["transitions"])

        agent = self.algo(self.hyperparameters)

        progress = tqdm(dataset, desc=f"Training {i}/{self.params.n}")

        for state, action, reward, next_state, done in dataset:
            agent.replay_buffer.push(state, action * 0.5, reward, next_state, done)

        for _ in progress:
            agent.update()
            agent.total_steps += 1
            training_step += 1

            if training_step % self.params.interval == 0:
                eval_results = self.evaluation(agent)
                id = training_step // self.params.interval

                with results_lock:
                    if id not in shared_results["rewards"]:
                        shared_results["rewards"][id] = eval_results
                    else:
                        current_results = shared_results["rewards"][id]
                        shared_results["rewards"][id] = current_results + eval_results

                    progress.set_postfix({"eval": self._get_stats(eval_results)})

        with results_lock:
            shared_results["agent"] = agent.get_weights()
>>>>>>> 828bb0aa7f23412cbf42a703c2dfc477c7bc17e2
