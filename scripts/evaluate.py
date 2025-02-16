import gymnasium as gym # noqa

from afu.agents.ddpg import DDPG
from afu.agents.sac import SAC
from afu.agents.afu import AFU

from afu.agents.afu_perrin import AFUPerrin
from afu.experiments.off_policy import OffPolicy
from afu.experiments.on_policy import OnPolicy
from afu.experiments.test import NewExperiment
import argparse


def get_algorithm(name: str):
    algorithms = {"ddpg": DDPG, "sac": SAC, "afu": AFU, "afuperrin": AFUPerrin}
    return algorithms[name.lower()]


def get_env(name: str):
    envs = {
        "cartpole": "CartPoleContinuousStudy-v0",
    }
    return envs[name.lower()]


def get_experiment(name: str):
    experiments = {"onpolicy": OnPolicy, "offpolicy": OffPolicy, "test": NewExperiment}
    return experiments[name.lower()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL algorithms")
    parser.add_argument(
        "--algo",
        type=str,
        choices=["ddpg", "sac", "afu", "afuperrin"],
        required=True,
        help="Algorithm to train (DDPG, SAC, or AFU)",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["cartpole"],
        required=True,
        help="Environment (cartpole)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["onpolicy", "offpolicy", "test"],
        required=True,
        help="Experiment to run (onpolicy, offpolicy)",
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
        "--seed", type=int, required=False, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    algo = get_algorithm(args.algo)
    env_name = get_env(args.env)
    experiment = get_experiment(args.experiment)

    experiment(
        algo=algo,
        env_name=env_name,
        params={
            "n": args.run,
            "interval": args.interval,
            "total_steps": args.steps,
        },
        seed=args.seed,
    ).run()


if __name__ == "__main__":
    main()
