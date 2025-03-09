from .base import Experiment
from .off_policy import OffPolicy
from .on_policy import OnPolicy
# from tqdm import tqdm  # type: ignore


class OffToOnPolicy(Experiment):
    def run(self, i, shared_results, results_lock):
        # Run OffPolicy
        off_policy_agent = OffPolicy(self.hyperparameters)
        off_policy_data = off_policy_agent.run(i, shared_results, results_lock)

        # Run OnPolicy with OffPolicy data
        on_policy_agent = OnPolicy(off_policy_data)
        on_policy_agent.run(i, shared_results, results_lock, off_policy_data)
