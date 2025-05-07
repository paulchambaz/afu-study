import argparse
import matplotlib.pyplot as plt
import torch
import math
import pickle
import numpy as np
from pathlib import Path
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


def get_v_critic_value(agent, state):
    max_q = float("-inf")
    for i in range(100):
        action = torch.tensor([[((i / 100) * 2 - 1) * 2]])

        q_workspace = Workspace()
        q_workspace.set("env/env_obs", 0, state)
        q_workspace.set("action", 0, action)
        agent.critic(q_workspace, t=0)
        q_values = q_workspace.get("critic/q_value", 0)

        if q_values > max_q:
            max_q = q_values

    return max_q


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


def main():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.labelsize": 20,
            "font.size": 20,
            "legend.fontsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )

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

    width = 50
    height = 50

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
            agent.policy_network.sample_action(policy_workspace, t=0)
            # action = policy_workspace.get("action", 0)
            action = policy_workspace.get("mean", 0)
            # actions[j, i] = action
            actions[j, i] = torch.tanh(action)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Completely turn off the grid
    ax1.grid(False)
    ax2.grid(False)

    # Create the extent for proper axis scaling
    extent = [0, 2 * np.pi, -8, 8]  # [xmin, xmax, ymin, ymax]

    # First subplot: V-values - using imshow for better SVG compatibility
    im1 = ax1.imshow(
        v_values,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
    )
    ax1.set_title("V-Values")
    ax1.set_xlabel("Angle (radians)")
    ax1.set_ylabel("Velocity")
    fig.colorbar(im1, ax=ax1)

    # Second subplot: Actions - using imshow
    im2 = ax2.imshow(
        actions,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        interpolation="nearest",
    )
    ax2.set_title("Actions")
    ax2.set_xlabel("Angle (radians)")
    ax2.set_ylabel("Velocity")
    fig.colorbar(im2, ax=ax2)

    # Tight layout
    plt.tight_layout()

    # Save with specific SVG-friendly settings
    plt.savefig(
        "figure.svg", format="svg", bbox_inches="tight", transparent=True, dpi=300
    )



if __name__ == "__main__":
    main()
