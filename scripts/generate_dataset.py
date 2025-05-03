import argparse
import gymnasium as gym
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

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


def collect_episodes(env_name, agent, episodes=100, render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    action_space = env.unwrapped.get_action_space()

    # Create a simple list to store all transitions
    transitions = []

    total_rewards = []
    total_steps = 0

    for episode in tqdm(range(episodes)):
        observation, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done:
            action = agent.select_action(observation, evaluation=False)
            scaled_action = scale_action(action, action_space)

            next_observation, reward, terminated, truncated, _ = env.step(scaled_action)
            done = terminated or truncated

            transitions.append((observation, action, reward, next_observation, done))

            observation = next_observation
            episode_reward += reward
            step += 1
            total_steps += 1

        total_rewards.append(episode_reward)

    env.close()

    # Print summary statistics
    print(f"\nCollection completed. Total steps: {total_steps}")
    print(f"Average reward over {episodes} episodes: {np.mean(total_rewards):.4f}")
    print(f"Min reward: {np.min(total_rewards):.4f}")
    print(f"Max reward: {np.max(total_rewards):.4f}")
    print(f"Standard deviation: {np.std(total_rewards):.4f}")

    return transitions, {
        "total_steps": total_steps,
        "num_episodes": episodes,
        "avg_reward": float(np.mean(total_rewards)),
        "min_reward": float(np.min(total_rewards)),
        "max_reward": float(np.max(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "rewards": total_rewards,
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
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes to run"
    )
    parser.add_argument(
        "--output", type=str, default="expert_data.pkl", help="Output file path"
    )

    # Weights and results loading
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

    # Collect episodes and save to file
    transitions, stats = collect_episodes(
        env_name=env_name, agent=agent, episodes=args.episodes
    )

    # Save transitions to pickle file
    output_path = Path(args.output)
    dataset = {
        "transitions": transitions,
        "env_name": env_name,
        "algo": args.algo,
        "stats": stats,
        "hyperparameters": hyperparameters if args.results else None,
    }

    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(
        f"Successfully saved dataset with {len(transitions)} transitions to {output_path}"
    )


if __name__ == "__main__":
    main()
