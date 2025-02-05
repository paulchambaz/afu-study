import gymnasium as gym
import numpy as np
import afu
from afu.agents.dqn import DQN
from afu.agents.ddpg import DDPG


def demo_run(agent, env):
    env = gym.make(env, render_mode="human")
    observation, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(observation, evaluation=True)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        env.render()

    env.close()

    print(f"Demo completed on {env} with total reward: {total_reward}")


def dqn_demo() -> None:
    params = {
        "env_name": "MountainCar-v0",  # Which environment to train on
        "hidden_size": [128, 128],  # Two hidden layers of size 126 each
        "learning_rate": 1e-3,  # Standard learning rate for Adam optimizer
        "batch_size": 128,  # How many transitions to sample for each update
        "replay_size": 100_000,  # Maximum transitions to store in replay buffer
        "target_update": 1000,  # Update target network every N steps
        "gamma": 0.99,  # Standard discount factor for RL
        "epsilon_start": 1.0,  # Start with 100% random actions
        "epsilon_end": 0.05,  # End with 5% random actions
        "epsilon_decay": 5000,  # Decay exploration over this many steps
        "max_episodes": 1000,  # Maximum number of episodes to train
        "max_steps": 500,  # Maximum steps per episode
    }

    agent = DQN(params)
    metrics = agent.train()

    final_avg_reward = np.mean(metrics["episode_rewards"][-100:])
    print(f"Training completed. Final average reward: {final_avg_reward:.2f}")

    save_path = "trained_dqn_agent_cartpole.pt"
    agent.save(save_path)

    print("\nRunning demonstrations...")
    for i in range(2):
        print(f"\nDemo {i+1}:")
        demo_run(agent, "CartPole-v1")

    loaded_agent = DQN.load_agent(save_path)
    for _ in range(3):
        demo_run(loaded_agent, "CartPole-v1")


def ddpg_demo() -> None:
    params = {
        "env_name": "CartPoleContinuousStudy-v0",  # Which environment to train on
        "actor_hidden_size": [
            128,
            128,
        ],  # Two hidden layers of size 128 each for the actor
        "critic_hidden_size": [
            128,
            128,
        ],  # Two hidden layers of size 128 each for the critic
        "actor_lr": 1e-3,  # Standard learning rate for Adam optimizer for the actor
        "critic_lr": 1e-3,  # Standard learning rate for Adam optimizer for the critic
        "batch_size": 128,  # How many transitions to sample for each update
        "replay_size": 100_000,  # Maximum transitions to store in replay buffer
        "tau": 0.005,  # Soft update parameter
        "gamma": 0.99,  # Standard discount factor for RL
        "noise_std": 0.1,  # Gaussian noise standard deviation
        "max_episodes": 1000,  # Maximum number of episodes to train
        "max_steps": 500,  # Maximum steps per episode
    }

    agent = DDPG(params)
    metrics = agent.train()

    final_avg_reward = np.mean(metrics["episode_rewards"][-100:])
    print(f"Training completed. Final average reward: {final_avg_reward:.2f}")

    save_path = "trained_dqn_agent_mountaincar.pt"
    agent.save(save_path)

    print("\nRunning demonstrations...")
    for i in range(2):
        print(f"\nDemo {i+1}:")
        demo_run(agent, "CartPoleContinuousStudy-v0")

    loaded_agent = DQN.load_agent(save_path)
    for _ in range(3):
        demo_run(loaded_agent, "CartPoleContinuousStudy-v0")


def main() -> None:
    # dqn_demo()
    ddpg_demo()


if __name__ == "__main__":
    main()
