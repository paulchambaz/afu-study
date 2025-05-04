from .base import Experiment
from tqdm import tqdm
import pickle


class Offline(Experiment):
    def run(self, i, shared_results, results_lock, manager):
        training_step = 0

        dataset_files = [
            "dataset/OnPolicy-SAC-PendulumStudy-v0-data.pk",
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
