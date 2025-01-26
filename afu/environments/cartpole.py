import gymnasium as gym  # type: ignore
from bbrl.agents import GymAgent  # type: ignore


def make_env():
    return gym.make("CartPole-v1")


def create_env_agent():
    env = make_env()
    return GymAgent(env, autoreset=True)
