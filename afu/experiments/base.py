import gymnasium as gym
from abc import ABC, abstractmethod
import pickle
import time
import random
import numpy as np
from pathlib import Path


class Experiment(ABC):
    def __init__(self, algo, env_name, params, seed):
        self.algo = algo
        self.env_name = env_name
        self.params = params
        self.results = {
            "rewards": {},
            "metadata": {
                "start_time": None,
                "end_time": None,
                "total_steps": 0,
                "seed": seed if seed is not None else self._generate_seed(),
            },
        }
        self._set_seeds()

        self.env = gym.make(self.env_name)
        self.observation_space = self.env.unwrapped.get_observation_space()
        self.action_space = self.env.unwrapped.get_action_space()

    def _generate_seed(self) -> int:
        return int(time.time() * 1000) % (2**32 - 1)

    def _set_seeds(self) -> None:
        seed = self.results["metadata"]["seed"]
        random.seed(seed)
        np.random.seed(seed)


    def _scale_action(self, action, source_space, target_space):
        source_low, source_high = (np.array([-1.0]), np.array([1.0]))  
        target_low, target_high = target_space
        
        normalized = (action - source_low) / (source_high - source_low)
        scaled = normalized * (target_high - target_low) + target_low
        return scaled

    def evaluation(self, agent, n=10):
        env = gym.make(self.env_name)
        env.reset(seed=self.results["metadata"]["seed"])
        results = []

        for _ in range(n):
            observation, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.select_action(observation, evaluation=True)
                action = self._scale_action(action, self.env.unwrapped.get_action_space())
                observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward

            results.append(total_reward)

        env.close()
        return results

    def log_metrics(self, step, metrics) -> None:
        for key, value in metrics.items():
            if key not in self.results:
                self.results[key] = {}
            self.results[key][step] = value

    def save_results(self) -> None:
        Path("results").mkdir(exist_ok=True)
        policy_type = self.__class__.__name__
        algo_name = self.algo.__name__
        self.results["metadata"]["end_time"] = time.time()
        filename = f"results/{policy_type}-{algo_name}-{self.env_name}.pk"
        with open(filename, "wb") as f:
            pickle.dump(self.results, f)

    def send_to_influxdb(self, metrics) -> None:
        """
        Placeholder for future InfluxDB integration
        This method would send metrics to InfluxDB for monitoring
        """
        pass

    @abstractmethod
    def run(self):
        self.results["metadata"]["start_time"] = time.time()
        pass

    def __del__(self):
        self.env.close()
