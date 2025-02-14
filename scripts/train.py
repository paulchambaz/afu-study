import gymnasium as gym
import numpy as np
from afu.agents.dqn import DQN
from afu.agents.ddpg import DDPG
from afu.agents.sac import SAC
from afu.agents.afu import AFU
from bbrl.workspace import Workspace  # type: ignore
from tqdm import tqdm  # type: ignore

# from afu_rljax.algorithm import AFU  # type: ignore
# from afu_rljax.trainer import Trainer  # type: ignore
# import jax
# from datetime import datetime
# import os


# def test_afu_cartpole():
#     env_id = "CartPoleContinuousStudy-v0"
#     env = gym.make(env_id)
#     env_test = gym.make(env_id)
#
#     algo = AFU(
#         num_agent_steps=50000,
#         # num_agent_steps=100000,
#         state_space=env.observation_space,
#         action_space=env.action_space,
#         seed=42,
#         tau=1e-2,
#         lr_actor=3e-4,
#         lr_critic=3e-4,
#         lr_alpha=3e-4,
#         units_actor=(128, 128),
#         units_critic=(128, 128),
#         gradient_reduction=0.8,
#         variant="alpha",
#         alg="AFU",
#     )
#
#     # Setup logging
#     time = datetime.now().strftime("%Y%m%d-%H%M")
#     log_dir = os.path.join("logs", f"cartpole_afu_{time}")
#     os.makedirs(log_dir, exist_ok=True)
#
#     # Create trainer and train
#     trainer = Trainer(
#         env=env,
#         env_test=env_test,
#         algo=algo,
#         # num_agent_steps=100000,
#         num_agent_steps=50000,
#         log_dir=log_dir,
#         eval_interval=1000,
#         seed=42,
#     )
#
#     print("Starting AFU training...")
#     trainer.train()
#
#     # save_path = "weights/trained_AFU_CartPoleContinuousStudy-v0.pt"
#     # os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     #
#     # # Get the parameters from the AFU object
#     # params_dict = {
#     #     'actor_params': algo.actor_state.params,
#     #     'critic_params': algo.critic_state.params,
#     #     'log_alpha': algo.log_alpha,
#     # }
#     #
#     # # Save parameters using numpy's savez
#     # with open(save_path, 'wb') as f:
#     #     jnp.savez(save_path, **params_dict)
#     #
#     # print(f"Saved AFU model parameters to {save_path}")
#
#     print("\nRunning demonstrations...")
#     for i in range(3):
#         print(f"\nDemo {i+1}:")
#         afu_demo_run(algo, env_id)
#
#
# def afu_demo_run(agent, env):
#     env = gym.make(env, render_mode="human")
#     observation, _ = env.reset()
#     done = False
#     total_reward = 0
#
#     while not done:
#         action = agent.select_action(observation)
#         observation, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated
#         total_reward += reward
#
#         env.render()
#
#     env.close()
#
#     print(f"Demo completed on {env} with total reward: {total_reward}")


# def demo_run(agent, env):
#     env = gym.make(env, render_mode="human")
#     observation, _ = env.reset()
#     done = False
#     total_reward = 0
#
#     while not done:
#         action = agent.select_action(observation, evaluation=True)
#         observation, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated
#         total_reward += reward
#
#         env.render()
#
#     env.close()
#
#     print(f"Demo completed on {env} with total reward: {total_reward}")


def demo_run(agent, env_name, render=True):
    """Run a demonstration episode."""
    env = gym.make(env_name, render_mode="human" if render else None)
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state, evaluation=True)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        if render:
            env.render()

    env.close()
    print(f"Demo episode reward: {total_reward}")
    return total_reward


def train_demo_v2(algo, env_name) -> None:
    params = {
        "env_name": env_name,
        "hidden_size": [128, 128],
        "learning_rate": 3e-4,
        "batch_size": 128,
        "replay_size": 10_000,
        "target_update": 1000,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 5000,
        "max_episodes": 500,
        "max_steps": 500,
    }

    # Initialize agent and workspace
    agent = algo(params)
    workspace = Workspace()

    # Training loop
    episode_rewards = []
    progress = tqdm(range(params["max_episodes"]), desc="Training")

    for episode in progress:
        state, _ = agent.env.reset()
        episode_reward = 0.0

        # Episode loop
        for step in range(params["max_steps"]):
            # Get action using our new agent system
            action = agent.get_action(state)

            # Execute action in environment
            next_state, reward, terminated, truncated, _ = agent.env.step(action)
            done = terminated or truncated

            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Perform update
            agent.update()

            state = next_state
            episode_reward += reward

            if done:
                break

        # Track rewards
        episode_rewards.append(episode_reward)

        if len(episode_rewards) >= 10:
            avg_reward = np.mean(episode_rewards[-10:])
            progress.set_postfix({"avg_reward": f"{avg_reward:.2f}"}, refresh=True)

    # Save results and demonstrate
    metrics = {"episode_rewards": episode_rewards}
    final_avg_reward = np.mean(metrics["episode_rewards"][-100:])
    print(f"Training completed. Final average reward: {final_avg_reward:.2f}")

    save_path = f"weights/trained_{env_name}.pt"
    agent.save(save_path)

    print("\nRunning demonstrations...")
    for i in range(1):
        print(f"\nDemo {i + 1}:")
        demo_run(agent, env_name)

    loaded_agent = algo.load_agent(save_path)
    for _ in range(1):
        demo_run(loaded_agent, env_name)


def train_demo(algo, env_name) -> None:
    params = {
        "env_name": env_name,  # Which environment to train on
        "hidden_size": [128, 128],  # Two hidden layers of size 126 each
        "learning_rate": 3e-4,  # Standard learning rate for Adam optimizer
        "batch_size": 128,  # How many transitions to sample for each update
        "replay_size": 10_000,  # Maximum transitions to store in replay buffer
        "target_update": 1000,  # Update target network every N steps
        "gamma": 0.99,  # Standard discount factor for RL
        "epsilon_start": 1.0,  # Start with 100% random actions
        "epsilon_end": 0.05,  # End with 5% random actions
        "epsilon_decay": 5000,  # Decay exploration over this many steps
        "max_episodes": 500,  # Maximum number of episodes to train
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
        "actor_lr": 3e-4,  # Standard learning rate for Adam optimizer for the actor
        "critic_lr": 3e-4,  # Standard learning rate for Adam optimizer for the critic
        "tau": 0.01,  # Soft update parameter
        "policy_hidden_size": [
            128,
            128,
        ],  # Two hidden layers for the policy network
        "q_hidden_size": [128, 128],  # Two hidden layers for the Q networks
        "policy_lr": 3e-4,  # Learning rate for policy
        "q_lr": 3e-4,  # Learning rate for Q networks
        "alpha_lr": 3e-4,  # Learning rate for temperature parameter
        "log_std_min": -20.0,  # Minimum log standard deviation for Gaussian policy
        "log_std_max": 2.0,  # Maximum log standard deviation for Gaussian policy
        "gradient_reduction": 0.8,  # Gradient reduction coefficient for AFU
    }

    agent = algo(params)
    metrics = agent.train()

    final_avg_reward = np.mean(metrics["episode_rewards"][-100:])
    print(f"Training completed. Final average reward: {final_avg_reward:.2f}")

    save_path = f"weights/trained_{env_name}.pt"
    agent.save(save_path)

    print("\nRunning demonstrations...")
    for i in range(1):
        print(f"\nDemo {i + 1}:")
        demo_run(agent, env_name)

    loaded_agent = algo.load_agent(save_path)
    for _ in range(1):
        demo_run(loaded_agent, env_name)


def main() -> None:
    train_demo_v2(DQNAgent, "CartPole-v1")
    # train_demo(DQN, "CartPole-v1")
    # train_demo(SAC, "CartPoleContinuousStudy-v0")
    # test_afu_cartpole()


if __name__ == "__main__":
    main()
