import gymnasium as gym  # noqa

from afu.agents.ddpg import DDPG
from afu.agents.sac import SAC
from afu.agents.afu import AFU
from afu.agents.afu_perrin import AFUPerrin

from afu.experiments.base import Experiment
from afu.experiments.off_policy import OffPolicy
from afu.experiments.on_policy import OnPolicy
from afu.experiments.test import NewExperiment

import argparse
from typing import Type, Any
from multiprocessing import Process, cpu_count
from itertools import product

# Reuse your existing configuration dictionaries
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
}

EXPERIMENTS: dict[str, Type[Experiment]] = {
    "onpolicy": OnPolicy,
    "offpolicy": OffPolicy,
    "test": NewExperiment,
}


def run_single_experiment(config: tuple[str, str, str, dict]) -> None:
    algo_name, env_key, experiment_name, params = config

    algo = ALGORITHMS[algo_name]
    env_name = ENVS[env_key]
    experiment = EXPERIMENTS[experiment_name]

    print(f"Starting experiment: {algo_name} on {env_key} with {experiment_name}")

    experiment(
        algo=algo, env_name=env_name, params=params, seed=params.get("seed")
    ).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RL algorithms")
    parser.add_argument(
        "--algo",
        type=str,
        choices=ALGORITHMS.keys(),
        required=False,
        help="Filter by algorithm",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=ENVS.keys(),
        required=False,
        help="Filter by environment",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=EXPERIMENTS.keys(),
        required=False,
        help="Filter by experiment type",
    )
    parser.add_argument(
        "--run",
        type=int,
        required=False,
        default=15,
        help="Number of runs for each experience",
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
        help="Max number of steps for each experiment",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Filter the configurations based on provided arguments
    algos = [args.algo] if args.algo else ALGORITHMS.keys()
    envs = [args.env] if args.env else ENVS.keys()
    experiments = [args.experiment] if args.experiment else EXPERIMENTS.keys()

    params = {
        "n": args.run,
        "interval": args.interval,
        "total_steps": args.steps,
        "seed": args.seed,
    }

    configurations = list(product(algos, envs, experiments, [params]))

    processes: list[Process] = []

    print(f"Running {len(configurations)} experiments in parallel")

    for config in configurations:
        p = Process(target=run_single_experiment, args=(config,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
