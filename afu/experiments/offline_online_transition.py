from .base import Experiment
import pickle
from tqdm import tqdm


class OfflineOnlineTransition(Experiment):
    def run(self, i, shared_results, results_lock, manager):
        training_step = 0

        dataset_files = [
            "dataset/OffPolicyDataset-AFU-PendulumStudy-v0-dataset.pk",
            "dataset/OnPolicyDataset-SAC-PendulumStudy-v0-dataset.pk",
        ]

        self.hyperparameters["dataset_files"] = dataset_files

        dataset = []
        for file in dataset_files:
            with open(file, "rb") as f:
                results = pickle.load(f)
                dataset.extend(results)

        agent = self.algo(self.hyperparameters)

        progress = tqdm(dataset, desc=f"Offline Training {i}/{self.params.n}")

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

                break

        progress = tqdm(
            range(self.params.total_steps),
            desc=f"Online Training {i}/{self.params.n}",
        )

        while training_step < self.params.total_steps:
            state, _ = agent.train_env.reset()

            while True:
                action = agent.select_action(state)
                scaled_action = self._scale_action(action, self.action_space)

                next_state, reward, terminated, truncated, _ = agent.train_env.step(
                    scaled_action
                )
                done = terminated or truncated

                agent.replay_buffer.push(state, action, reward, next_state, done)

                agent.update()

                state = next_state
                agent.total_steps += 1
                training_step += 1

                progress.update(1)

                if training_step % self.params.interval == 0:
                    eval_results = self.evaluation(agent)
                    id = training_step // self.params.interval

                    with results_lock:
                        if id not in shared_results["rewards"]:
                            shared_results["rewards"][id] = eval_results
                        else:
                            current_results = shared_results["rewards"][id]
                            shared_results["rewards"][id] = (
                                current_results + eval_results
                            )

                    progress.set_postfix({"eval": self._get_stats(eval_results)})

                if done:
                    break

                if training_step >= self.params.total_steps:
                    break

            if training_step >= self.params.total_steps:
                break

        with results_lock:
            shared_results["agent"] = agent.get_weights()
