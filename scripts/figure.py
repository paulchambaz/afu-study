import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch
import math
import gymnasium as gym
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from bbrl.workspace import Workspace  # type: ignore

from afu.agents.ddpg import DDPG
from afu.agents.sac import SAC
from afu.agents.afu import AFU
from afu.agents.calql import CALQL
from afu.agents.iql import IQL

ALGORITHMS = {
    "ddpg": DDPG,
    "sac": SAC,
    "afu": AFU,
    "calql": CALQL,
    "iql": IQL,
}

ENVS = {
    "cartpole": "CartPoleContinuousStudy-v0",
    "pendulum": "PendulumStudy-v0",
    "lunarlander": "LunarLanderContinuousStudy-v0",
    "swimmer": "SwimmerStudy-v0",
    "ant": "AntStudy-v0",
    "bipedalwalker": "BipedalWalkerStudy-v0",
}


def load_agent_from_weights(env_name, algo_class, weights_path, hyperparameters=None):
    if hyperparameters is None:
        # Use default hyperparameters if none provided
        hyperparameters = algo_class._get_params_defaults()
        hyperparameters["env_name"] = env_name

    agent = algo_class(hyperparameters)
    agent.load(weights_path)
    print(f"Loaded weights from {weights_path}")
    return agent


def load_results_file(results_path):
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Collect expert experience data from trained agent"
    )

    # Basic arguments
    parser.add_argument(
        "--algo",
        type=str,
        choices=ALGORITHMS.keys(),
        required=True,
        help="Algorithm to use",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=ENVS.keys(),
        required=True,
        help="Environment to run",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--weights", type=str, help="Path to weights file to load")
    group.add_argument(
        "--results",
        type=str,
        help="Path to results file (will also load associated weights)",
    )

    args = parser.parse_args()

    algo_class = ALGORITHMS[args.algo]
    env_name = ENVS[args.env]

    if args.results:
        results = load_results_file(args.results)
        hyperparameters = results["hyperparameter"]

        results_path = Path(args.results)
        weights_path = results_path.parent / f"{results_path.stem}-weights.pt"

        if not weights_path.exists():
            print(f"Warning: Could not find weights file at {weights_path}")
            print("Attempting to use algorithm defaults instead.")
            agent = algo_class(hyperparameters)
        else:
            agent = load_agent_from_weights(
                env_name, algo_class, weights_path, hyperparameters
            )
    else:
        agent = load_agent_from_weights(env_name, algo_class, args.weights)

    width = 100
    height = 100

    angles = np.zeros((width))
    velocities = np.zeros((height))
    v_values = np.zeros((height, width))
    actions = np.zeros((height, width))

    for i in range(width):
        for j in range(height):
            theta = 2 * math.pi * i / float(width)
            angles[i] = theta
            velocity = ((j / float(height)) * 2 - 1) * 8
            velocities[j] = velocity

            x = math.cos(theta)
            y = math.sin(theta)

            state = torch.tensor([[x, y, velocity]])

            v_value = get_v_q_value(agent, state)
            v_values[j, i] = v_value

            policy_workspace = Workspace()
            policy_workspace.set("env/env_obs", 0, state)
            agent.policy_network(policy_workspace, t=0)
            # agent.policy_network.sample_action(policy_workspace, t=0)
            action = policy_workspace.get("mean", 0)
            actions[j, i] = torch.tanh(action)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("CALQL", fontsize=16)

    # First subplot: V-values
    im1 = ax1.pcolormesh(angles, velocities, v_values, shading="auto", cmap="viridis")
    ax1.set_title("V-Values as a function of angle and velocity")
    ax1.set_xlabel("Angle (radians)")
    ax1.set_ylabel("Velocity")
    fig.colorbar(im1, ax=ax1, label="V Value")

    # Second subplot: Actions
    im2 = ax2.pcolormesh(angles, velocities, actions, shading="auto", cmap="coolwarm")
    ax2.set_title("Actions as a function of angle and velocity")
    ax2.set_xlabel("Angle (radians)")
    ax2.set_ylabel("Velocity")
    fig.colorbar(im2, ax=ax2, label="Action")

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title
    plt.show()


def get_v12_value(agent, state):
    v1_workspace = Workspace()
    v1_workspace.set("env/env_obs", 0, state)
    agent.v_network1(v1_workspace, t=0)
    v1_values = v1_workspace.get("v1/v_value", 0)

    v2_workspace = Workspace()
    v2_workspace.set("env/env_obs", 0, state)
    agent.v_network2(v2_workspace, t=0)
    v2_values = v2_workspace.get("v2/v_value", 0)

    v_value = min(v1_values, v2_values)

    return v_value


def get_v_value(agent, state):
    v_workspace = Workspace()
    v_workspace.set("env/env_obs", 0, state)
    agent.v_network(v_workspace, t=0)
    v_value = v_workspace.get("v/v_value", 0)

    return v_value


def get_v_q_value(agent, state):
    max_q = float("-inf")
    for i in range(100):
        action = torch.tensor([[((i / 100) * 2 - 1) * 2]])

        q_workspace = Workspace()
        q_workspace.set("env/env_obs", 0, state)
        q_workspace.set("action", 0, action)
        agent.q_network(q_workspace, t=0)
        q_values = q_workspace.get("q/q_value", 0)

        if q_values > max_q:
            max_q = q_values

    return max_q


if __name__ == "__main__":
    main()
