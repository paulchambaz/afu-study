import gymnasium as gym
import torch
import optuna
import multiprocessing as mp
from abc import ABC, abstractmethod
import pickle
import time
import random
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf


class Experiment(ABC):
    def __init__(self, algo, **kwargs):
        self.algo = algo

        self.params = OmegaConf.merge(
            self._get_params_defaults(),
            OmegaConf.create(kwargs),
        )

        self.hyperparameters = algo._get_params_defaults()
        self.hyperparameters["env_name"] = self.params.env_name

        seed = (
            self.params.seed if hasattr(self.params, "seed") else self._generate_seed()
        )
        self.results = {
            "rewards": {},
            "params": self.params,
            "hyperparameter": self.hyperparameters,
            "metadata": {
                "total_steps": 0,
                "seed": seed,
            },
        }

        self._set_seeds()

        self.env = gym.make(self.params.env_name)
        self.observation_space = self.env.unwrapped.get_observation_space()
        self.action_space = self.env.unwrapped.get_action_space()

    def get_score(self) -> float:
        rewards = sorted(self.results["rewards"].keys())
        points = rewards[int(0.9 * len(rewards)) :]
        results = []
        for point in points:
            results.extend(self.results["rewards"][point])
        min_val, q1, iqm, q3, max_val = self._compute_stats(results)
        return (q1 + iqm + q3) / 3

    def _compute_stats(self, data):
        data = np.array(data)
        min_val = np.min(data)
        max_val = np.max(data)
        q1, q3 = np.percentile(data, [25, 75])
        mask = (data >= q1) & (data <= q3)
        iqm = np.mean(data[mask])
        return min_val, q1, iqm, q3, max_val

    def _get_stats(self, data):
        min_val, q1, iqm, q3, max_val = self._compute_stats(data)
        return f"[{min_val:.1f}|{q1:.1f}|{iqm:.1f}|{q3:.1f}|{max_val:.1f}]"

    def _generate_seed(self) -> int:
        return int(time.time() * 1000) % (2**32 - 1)

    def _set_seeds(self) -> None:
        seed = self.results["metadata"]["seed"]
        random.seed(seed)
        np.random.seed(seed)

    def _scale_action(self, action, target_space):
        source_low, source_high = -1.0, 1.0
        target_low, target_high = target_space

        action = np.clip(action, source_low, source_high)

        normalized = (action - source_low) / (source_high - source_low)
        scaled = normalized * (target_high - target_low) + target_low

        scaled = np.clip(scaled, target_low, target_high)

        return scaled

    def evaluation(self, agent, n=10):
        env = gym.make(self.params.env_name)

        results = []
        transitions = []

        for _ in range(n):
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.select_action(state, evaluation=True)
                scaled_action = self._scale_action(action, self.action_space)
                next_state, reward, terminated, truncated, _ = env.step(scaled_action)
                done = terminated or truncated

                transitions.append((state, action, reward, next_state, done))

                total_reward += reward
                state = next_state

            results.append(total_reward)

        env.close()
        return results, transitions

    def log_metrics(self, step, metrics) -> None:
        for key, value in metrics.items():
            if key not in self.results:
                self.results[key] = {}
            self.results[key][step] = value

    def save_results(self) -> None:
        Path("results").mkdir(exist_ok=True)
        Path("weights").mkdir(exist_ok=True)
        Path("dataset").mkdir(exist_ok=True)
        policy_type = self.__class__.__name__
        algo_name = self.algo.__name__

        # Save experiment results
        results_filename = (
            f"results/{policy_type}-{algo_name}-{self.params.env_name}.pk"
        )
        with open(results_filename, "wb") as f:
            pickle.dump(self.results, f)
        print(f"Results saved to {results_filename}")

        # Save agent weights
        weights_filename = (
            f"weights/{policy_type}-{algo_name}-{self.params.env_name}-weights.pt"
        )
        torch.save(self.agent, weights_filename)
        print(f"Agent weights saved to {weights_filename}")

        dataset_filename = (
            f"dataset/{policy_type}-{algo_name}-{self.params.env_name}-dataset.pk"
        )
        with open(dataset_filename, "wb") as f:
            pickle.dump(self.transitions, f)
        print(f"Results saved to {dataset_filename}")

    def send_to_influxdb(self, metrics) -> None:
        """
        Placeholder for future InfluxDB integration
        This method would send metrics to InfluxDB for monitoring
        """
        pass

    @abstractmethod
    def run(self, id):
        pass

    def run_parallel(self, n_runs=None):
        if n_runs is None:
            n_runs = self.params.n

        manager = mp.Manager()
        shared_results = manager.dict()
        for key, value in self.results.items():
            shared_results[key] = value.copy()
        shared_results["rewards"] = manager.dict()

        results_lock = mp.Lock()

        processes = []
        for i in range(n_runs):
            p = mp.Process(
                target=self.run, args=(i, shared_results, results_lock, manager)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        self.results = {"rewards": dict(shared_results["rewards"])}
        self.agent = dict(shared_results["agent"])
        self.transitions = shared_results["transitions"]

    def tuned_run(self, n_trials=50, n_parallel_trials=5, n_runs=3):
        if n_trials > 0:

            def objective(trial):
                params = {}
                params["env_name"] = self.params.env_name
                for name, (
                    param_type,
                    min_val,
                    max_val,
                    log_scale,
                ) in self.algo._get_hp_space().items():
                    if param_type == "float":
                        params[name] = trial.suggest_float(
                            name, min_val, max_val, log=log_scale
                        )
                    elif param_type == "int":
                        params[name] = trial.suggest_int(
                            name, min_val, max_val, log=log_scale
                        )
                    else:
                        print("Wrong param type for trial")
                        exit(1)

                experiment = self.__class__(
                    algo=self.algo,
                    env_name=self.params.env_name,
                    n=3,
                    interval=self.params.interval,
                    total_steps=self.params.total_steps,
                )
                experiment.hyperparameter = OmegaConf.create(params)
                experiment.run_parallel(n_runs)
                score = experiment.get_score()
                print(f"Score: {score}")
                return score

            sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=10)
            study = optuna.create_study(direction="maximize", sampler=sampler)

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(objective, n_trials=n_trials, n_jobs=n_parallel_trials)

            print(f"Found the best parameters after {n_trials} trials")
            best_params = study.best_params
            print(best_params)

            best_params["env_name"] = self.params.env_name

            self.hyperparameters = OmegaConf.create(best_params)
            self.hyperparameters["env_name"] = self.params.env_name
            self.results["hyperparameter"] = self.hyperparameters

        self.run_parallel()
        self.save_results()

    @classmethod
    def _get_params_defaults(cls) -> OmegaConf:
        return OmegaConf.create(
            {
                "n": 15,
                "interval": 100,
                "update_interval": 1000,
                "batch_size": 50_000,
                "offline_steps": 500,
                "offline_interval": 10_000,
                "total_steps": 50_000,
                "seed": None,
            }
        )

    def __del__(self):
        self.env.close()
