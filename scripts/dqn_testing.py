import gymnasium as gym
import argparse
import torch
from omegaconf import OmegaConf
from pathlib import Path
import pickle

# Import your DQN implementation
from afu.agents.dqn import DQN  # Assuming the DQN class is in dqn.py


def run_dqn_cartpole(
    episodes=1000,
    max_steps=500,
    evaluation_interval=100,
    save_path="results/dqn_cartpole",
):
    # Environment setup
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    # Configure DQN hyperparameters
    hyperparameters = OmegaConf.create(
        {
            "env_name": env_name,
            "hidden_size": 64,
            "q_lr": 1e-3,
            "replay_size": 10000,
            "batch_size": 64,
            "gamma": 0.99,
            "decay_rate": 0.99,
            "target_update": 10,
        }
    )

    # Initialize DQN agent
    agent = DQN(hyperparameters)

    # Training loop
    total_steps = 0
    rewards_history = {}

    print(f"Starting DQN training on {env_name}")

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done and step < max_steps:
            # Select action
            action = agent.select_action(state)

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store experience in replay buffer
            # Note: Using the correct structure to match your update pattern
            # Creating a two-timestep transition with t=0 (current) and t=1 (next)
            transition = {
                "env/env_obs": torch.tensor(
                    [[state], [next_state]], dtype=torch.float32
                ),
                "action": torch.tensor(
                    [[action], [0]], dtype=torch.long
                ),  # Next action doesn't matter
                "env/reward": torch.tensor(
                    [[0], [reward]], dtype=torch.float32
                ),  # Reward at t+1
                "env/done": torch.tensor([[False], [done]], dtype=torch.bool),
            }
            agent.replay_buffer.put(transition)

            # Update agent
            agent.update()

            # Move to next state
            state = next_state
            episode_reward += reward
            step += 1
            total_steps += 1

        # Log episode results
        if episode % 10 == 0:
            print(
                f"Episode {episode}/{episodes} | Steps: {step} | Reward: {episode_reward:.1f} | Epsilon: {agent.epsilon:.3f}"
            )

        # Evaluate agent periodically
        if episode % evaluation_interval == 0:
            eval_rewards = evaluate_agent(agent, env, n_episodes=5)
            avg_reward = sum(eval_rewards) / len(eval_rewards)
            rewards_history[episode] = eval_rewards
            print(f"Evaluation at episode {episode}: Average reward: {avg_reward:.1f}")

    # Save results
    Path(save_path).mkdir(parents=True, exist_ok=True)
    results = {
        "rewards": rewards_history,
        "hyperparameters": hyperparameters,
        "total_steps": total_steps,
    }

    with open(f"{save_path}/results.pkl", "wb") as f:
        pickle.dump(results, f)

    # Save model
    agent.save(f"{save_path}/model.pt")

    print(f"Training complete. Results saved to {save_path}")
    env.close()
    return agent


def evaluate_agent(agent, env, n_episodes=5):
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, evaluation=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        rewards.append(episode_reward)

    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on CartPole")
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of episodes to train"
    )
    parser.add_argument(
        "--eval-interval", type=int, default=100, help="Evaluation interval"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="results/dqn_cartpole",
        help="Path to save results",
    )

    args = parser.parse_args()

    run_dqn_cartpole(
        episodes=args.episodes,
        evaluation_interval=args.eval_interval,
        save_path=args.save_path,
    )
