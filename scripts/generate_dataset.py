import argparse
import gymnasium as gym
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path

from afu.agents.ddpg import DDPG
from afu.agents.sac import SAC
from afu.agents.afu import AFU
from afu.agents.afu_perrin import AFUPerrin

ALGORITHMS = {
    "ddpg": DDPG,
    "sac": SAC,
    "afu": AFU,
    "afuperrin": AFUPerrin,
}

ENVS = {
    "cartpole": "CartPoleContinuousStudy-v0",
    "pendulum": "PendulumStudy-v0",
    "lunarlander": "LunarLanderContinuousStudy-v0",
    "swimmer": "SwimmerStudy-v0",
    "ant": "AntStudy-v0",
    "bipedalwalker": "BipedalWalkerStudy-v0",
}


def scale_action(action, target_space):
    source_low, source_high = -1.0, 1.0
    target_low, target_high = target_space

    action = np.clip(action, source_low, source_high)

    normalized = (action - source_low) / (source_high - source_low)
    scaled = normalized * (target_high - target_low) + target_low

    scaled = np.clip(scaled, target_low, target_high)

    return scaled


def generate_dataset(env_name, agent, suffix="", episodes=5):
    env = gym.make(env_name)
    action_space = env.unwrapped.get_action_space()
    print(action_space)

    total_rewards = []
    dataset = []

    for episode in tqdm(range(episodes)):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done:
            action = agent.select_action(state, evaluation=True)
            scaled_action = scale_action(action, action_space)

            next_state, reward, terminated, truncated, _ = env.step(scaled_action)
            done = terminated or truncated
            episode_reward += reward

            transition = (state, action, reward, next_state, done)
            # print(transition)
            dataset.append(transition)

            state = next_state

            step += 1

        total_rewards.append(episode_reward)

    env.close()

    agent_name = agent.__class__.__name__
    save_path = Path(f"dataset/{agent_name}-{env_name}-{suffix}-data.pk")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(len(dataset))

    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"Dataset saved to {save_path}")


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
        description="Visualize trained agent in environment"
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
        help="Environment to visualize",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to run"
    )

    # Weights and results loading
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--weights", type=str, help="Path to weights file to load")
    group.add_argument(
        "--results",
        type=str,
        help="Path to results file (will also load associated weights)",
    )

    parser.add_argument(
        "--suffix",
        type=str,
        help="Suffix for the name of the file saved",
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

    # Run visualization
    generate_dataset(
        env_name=env_name,
        agent=agent,
        episodes=args.episodes,
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
