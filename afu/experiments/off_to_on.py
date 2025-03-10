from .base import Experiment
from .off_policy import OffPolicy
from .on_policy import OnPolicy
# from tqdm import tqdm  # type: ignore


class OffToOnPolicy(Experiment):
    def run(self, i, shared_results, results_lock):
        # Run OffPolicy
        off_policy_agent = OffPolicy(
            algo=self.algo,
            env_name=self.params.env_name,
            n=self.hyperparameters.get("run", 1),
            interval=self.hyperparameters.get("interval", 1),
            total_steps=self.hyperparameters.get("steps", 1000),
        )
        off_policy_agent.run(i, shared_results, results_lock)

        algo_name = self.hyperparameters.get("algo", "AFUPerrin")
        self.algo.load(f"results/OffPolicy-{algo_name}-{self.params.env_name}.pt", "rb")

        # Run OnPolicy with OffPolicy data
        on_policy_agent = OnPolicy(
            algo=self.algo,
            env_name=self.params.env_name,
            n=self.hyperparameters.get("run", 1),
            interval=self.hyperparameters.get("interval", 1),
            total_steps=self.hyperparameters.get("steps", 1000),
        )
        on_policy_agent.run(i, shared_results, results_lock)
