from .base import Experiment
from tqdm import tqdm  # type: ignore


class OnPolicy(Experiment):
    def run(self, i, shared_results, results_lock):
        training_steps = 0
        agent = self.algo(self.hyperparameters)

        progress = tqdm(
            range(self.params.total_steps),
            desc=f"Training {i}/{self.params.n}",
        )

        while training_steps < self.params.total_steps:
            state, _ = agent.train_env.reset()

            while True:
                action = agent.select_action(state)
                action = self._scale_action(action, self.action_space)

                next_state, reward, terminated, truncated, _ = agent.train_env.step(
                    action
                )
                done = terminated or truncated

                agent.replay_buffer.push(state, action, reward, next_state, done)

                # print(f"before update {training_steps}")
                agent.update()
                # print(f"after update {training_steps}")

                state = next_state
                agent.total_steps += 1
                training_steps += 1

                progress.update(1)

                if training_steps % self.params.interval == 0:
                    eval_results = self.evaluation(agent)
                    id = training_steps // self.params.interval

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

                if training_steps >= self.params.total_steps:
                    break

            if training_steps >= self.params.total_steps:
                break

        with results_lock:
            shared_results["agent"] = agent.get_weights()
