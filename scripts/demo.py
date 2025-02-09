from afu.agents.ddpg import DDPG
from afu.agents.dqn import DQN
from afu.agents.sac import SAC
import gymnasium as gym
from afu_rljax.algorithm import AFU  # type: ignore
import jax.numpy as jnp
import pickle


def afu_demo_run(agent, env):
    env = gym.make(env, render_mode="human")
    observation, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        env.render()

    env.close()


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


def main() -> None:
    # env_name = "CartPole-v1"
    # dqn_agent = DQN.load_agent(f"weights/trained_{env_name}.pt")
    # demo_run(dqn_agent, env_name)
    #
    # env_name = "MountainCar-v0"
    # dqn_agent = DQN.load_agent(f"weights/trained_{env_name}.pt")
    # demo_run(dqn_agent, env_name)

    env_name = "CartPoleContinuousStudy-v0"
    ddpg_agent = SAC.load_agent(f"weights/trained_{env_name}.pt")
    demo_run(ddpg_agent, env_name)

    # env_name = "MountainCarContinuousStudy-v0"
    # ddpg_agent = SAC.load_agent(f"weights/trained_{env_name}.pt")
    # demo_run(ddpg_agent, env_name)

    # env_name = "CartPoleContinuousStudy-v0"
    # afu_agent = load_afu_params("weights/trained_AFU_CartPoleContinuousStudy-v0.npz", new_algo)
    # afu_demo_run(afu_agent, env_name)


if __name__ == "__main__":
    main()
