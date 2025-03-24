import gymnasium as gym  # noqa

from afu.agents.ddpg import DDPG
from afu.agents.sac import SAC
from afu.agents.afu import AFU

from afu.agents.afu_perrin import AFUPerrin

from afu.experiments.base import Experiment
from afu.experiments.off_policy import OffPolicy
from afu.experiments.on_policy import OnPolicy
from afu.experiments.off_to_on import OffToOnPolicy
from afu.experiments.random_walk_policy import RandomWalkPolicy
from afu.experiments.hybrid_policy import HybridPolicy
import argparse

from typing import Type, Any


ALGORITHMS: dict[str, Any] = {
    "ddpg": DDPG,
    "sac": SAC,
    "afu": AFU,
    "afuperrin": AFUPerrin,
}


ENVS: dict[str, str] = {
    "cartpole": "CartPoleContinuousStudy-v0",
    "pendulum": "PendulumStudy-v0",
    "lunarlander": "LunarLanderContinuousStudy-v0",
    "swimmer": "SwimmerStudy-v0",
    "ant": "AntStudy-v0",
    "bipedalwalker": "BipedalWalkerStudy-v0",
}

EXPERIMENTS: dict[str, Type[Experiment]] = {
    "onpolicy": OnPolicy,
    "offpolicy": OffPolicy,
    "offtoon": OffToOnPolicy,
    "randomwalkpolicy": RandomWalkPolicy,
    "hybridpolicy": HybridPolicy,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL algorithms")
    parser.add_argument(
        "--algo",
        type=str,
        choices=ALGORITHMS.keys(),
        required=True,
        help="Algorithm to train",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=ENVS.keys(),
        required=True,
        help="Environment",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=EXPERIMENTS.keys(),
        required=True,
        help="Experiment to run",
    )
    parser.add_argument(
        "--run",
        type=int,
        required=False,
        default=15,
        help="Number of runs for the experience",
    )
    parser.add_argument(
        "--interval",
        type=int,
        required=False,
        default=100,
        help="Number of steps between measure of performance",
    )
    parser.add_argument(
        "--steps",
        type=int,
        required=False,
        default=50_000,
        help="Max number of steps for the experiment",
    )
    parser.add_argument(
        "--trials",
        type=int,
        required=False,
        default=80,
        help="Number of trials for optuna",
    )
    parser.add_argument(
        "--seed", type=int, required=False, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    algo = ALGORITHMS[args.algo]
    env_name = ENVS[args.env]
    Experiment = EXPERIMENTS[args.experiment]

    experiment = Experiment(
        algo=algo,
        env_name=env_name,
        n=args.run,
        interval=args.interval,
        total_steps=args.steps,
        seed=args.seed,
    )

    experiment.tuned_run(
        n_trials=args.trials,
        n_parallel_trials=5,
        n_runs=3,
    )


if __name__ == "__main__":
    main()
