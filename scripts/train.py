import gymnasium as gym
import numpy as np
from afu.agents.dqn import DQN
from afu.agents.ddpg import DDPG

from afu_rljax.algorithm import AFU  # type: ignore
from afu_rljax.trainer import Trainer  # type: ignore
import jax


def test_afu_cartpole():
    env_id = "CartPoleContinuousStudy-v0"
    env = gym.make(env_id)
    env_test = gym.make(env_id)

    algo = AFU(
        num_agent_steps=1000,
        # num_agent_steps=100000,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=42,
        tau=1e-2,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        units_actor=(256, 256),
        units_critic=(256, 256),
        gradient_reduction=0.8,
        variant="alpha",
        alg="AFU",
    )

    # Create trainer and train
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        # num_agent_steps=100000,
        num_agent_steps=1000,
        eval_interval=1000,
        seed=42,
    )

    print("Starting AFU training...")
    trainer.train()

    print("\nRunning demonstrations...")
    for i in range(3):
        print(f"\nDemo {i+1}:")
        afu_demo_run(algo, env_id)


def afu_demo_run(agent, env):
    env = gym.make(env, render_mode="human")
    observation, _ = env.reset()
    done = False
    total_reward = 0

    rng = jax.random.PRNGKey(0)

    while not done:
        action = agent.select_action(observation, rng, eval=True)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        env.render()

    env.close()

    print(f"Demo completed on {env} with total reward: {total_reward}")


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


def train_demo(algo, env_name) -> None:
    params = {
        "env_name": env_name,  # Which environment to train on
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
        "noise_std": 0.1,  # Gaussian noise standard deviation
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
        "tau": 0.005,  # Soft update parameter
    }

    agent = algo(params)
    metrics = agent.train()

    final_avg_reward = np.mean(metrics["episode_rewards"][-100:])
    print(f"Training completed. Final average reward: {final_avg_reward:.2f}")

    save_path = f"weights/trained_{env_name}.pt"
    agent.save(save_path)

    print("\nRunning demonstrations...")
    for i in range(2):
        print(f"\nDemo {i+1}:")
        demo_run(agent, env_name)

    loaded_agent = algo.load_agent(save_path)
    for _ in range(3):
        demo_run(loaded_agent, env_name)


# def ddpg_demo() -> None:
#     # env_name = "MountainCarContinuousStudy-v0"
#     env_name = "CartPoleContinuousStudy-v0"
#     params = {
#         "env_name": env_name,  # Which environment to train on
#         "actor_hidden_size": [
#             128,
#             128,
#         ],  # Two hidden layers of size 128 each for the actor
#         "critic_hidden_size": [
#             128,
#             128,
#         ],  # Two hidden layers of size 128 each for the critic
#         "actor_lr": 1e-3,  # Standard learning rate for Adam optimizer for the actor
#         "critic_lr": 1e-3,  # Standard learning rate for Adam optimizer for the critic
#         "batch_size": 128,  # How many transitions to sample for each update
#         "replay_size": 100_000,  # Maximum transitions to store in replay buffer
#         "tau": 0.005,  # Soft update parameter
#         "gamma": 0.99,  # Standard discount factor for RL
#         "noise_std": 0.1,  # Gaussian noise standard deviation
#         "max_episodes": 1000,  # Maximum number of episodes to train
#         "max_steps": 500,  # Maximum steps per episode
#     }
#
#     agent = DDPG(params)
#     metrics = agent.train()
#
#     final_avg_reward = np.mean(metrics["episode_rewards"][-100:])
#     print(f"Training completed. Final average reward: {final_avg_reward:.2f}")
#
#     save_path = f"weights/trained_{env_name}.pt"
#     agent.save(save_path)
#
#     print("\nRunning demonstrations...")
#     for i in range(2):
#         print(f"\nDemo {i+1}:")
#         demo_run(agent, env_name)
#
#     loaded_agent = DDPG.load_agent(save_path)
#     for _ in range(3):
#         demo_run(loaded_agent, env_name)


def main() -> None:
    # train_demo(DQN, "MountainCar-v0")
    train_demo(DDPG, "CartPoleContinuousStudy-v0")
    # test_afu_cartpole()


if __name__ == "__main__":
    main()
