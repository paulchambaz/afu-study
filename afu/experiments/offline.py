from .base import Experiment
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
