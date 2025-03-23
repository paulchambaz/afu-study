import argparse
import gymnasium as gym
import pickle
import numpy as np
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


def visualize_environment(env_name, agent, episodes=5, render_mode="human"):
    env = gym.make(env_name, render_mode=render_mode)
    action_space = env.unwrapped.get_action_space()
    print(action_space)

    total_rewards = []

    for episode in range(episodes):
        observation, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0

        print(f"Episode {episode + 1}/{episodes}")

        while not done:
            env.render()
            action = agent.select_action(observation, evaluation=True)
            scaled_action = scale_action(action, action_space)

            observation, reward, terminated, truncated, _ = env.step(scaled_action)
            done = terminated or truncated
            episode_reward += reward

            print(f"Step {step}: Action={scaled_action}, Reward={reward:.4f}")

            step += 1

        print(f"Episode {episode + 1} finished with total reward: {episode_reward:.4f}")
        total_rewards.append(episode_reward)

    env.close()

    print(f"\nAverage reward over {episodes} episodes: {np.mean(total_rewards):.4f}")
    print(f"Min reward: {np.min(total_rewards):.4f}")
    print(f"Max reward: {np.max(total_rewards):.4f}")


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

    # Render mode
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "rgb_array"],
        help="Render mode for the environment",
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
    visualize_environment(
        env_name=env_name,
        agent=agent,
        episodes=args.episodes,
        render_mode=args.render_mode,
    )


if __name__ == "__main__":
    main()
