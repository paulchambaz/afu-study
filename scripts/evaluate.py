import gymnasium as gym
from tqdm import tqdm  # type: ignore
from pathlib import Path
import numpy as np
import pickle
from afu.agents.ddpg import DDPG
from afu.agents.sac import SAC
from afu.agents.afu import AFU


def evaluation(agent, env_name, n=15):
    env = gym.make(env_name)

    results = []

    for _ in range(n):
        observation, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(observation, evaluation=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        results.append(total_reward)

    env.close()

    return results


def off_policy_training_evaluate(algo, env_name, total_steps, interval, n=15):
    params = {
        "env_name": env_name,
        "actor_hidden_size": [128, 128],
        "critic_hidden_size": [128, 128],
        "noise_std": 0.1,
        "replay_size": 100_000,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "tau": 0.01,
        "gamma": 0.99,
        "batch_size": 128,
        "max_episodes": 500,
        "max_steps": 500,
        "policy_hidden_size": [128, 128],
        "q_hidden_size": [128, 128],
        "policy_lr": 3e-4,
        "q_lr": 3e-4,
        "alpha_lr": 3e-4,
        "hidden_size": [128, 128],
        "log_std_min": -20.0,
        "log_std_max": 2.0,
        "gradient_reduction": 0.8,
        "learning_rate": 3e-4,
    }

    training_rewards = {}

    for i in range(n):
        training_steps = 0
        agent = algo(params)
        env = gym.make(env_name)

        env_low = np.array(
            [
                -4.8,
                -8.0,
                -0.418,
                -8.0,
            ]
        ) / 2.0
        env_high = np.array(
            [
                4.8,
                8.0,
                0.418,
                8.0,
            ]
        ) / 2.0

        rewards = []
        progress = tqdm(range(total_steps), desc=f"Training {i}/{n}")

        for step in progress:
            agent.train_env.reset()
            random_state = np.random.uniform(low=env_low, high=env_high)
            agent.train_env.unwrapped.set_state(*random_state)

            action = np.random.uniform(low=-1.0, high=1.0, size=(1,))
            next_state, reward, terminated, truncated, _ = agent.train_env.step(
                action
            )
            done = terminated or truncated

            agent.replay_buffer.push(
                random_state, action, reward, next_state, done
            )
            agent.update()

            reward += float(reward)
            agent.total_steps += 1
            training_steps += 1

            rewards.append(reward)

            if len(rewards) >= 10:
                avg_reward = np.mean(rewards[-10:])
                progress.set_postfix(
                    {"avg_reward": f"{avg_reward:.2f}"}, refresh=True
                )

            if training_steps % interval == 0:
                results = evaluation(agent, env_name)
                id = training_steps / interval
                if id not in training_rewards:
                    training_rewards[id] = []
                training_rewards[id].extend(results)

        env.close()

    Path("results").mkdir(exist_ok=True)

    algo_name = algo.__name__
    filename = f"results/off-policy-{algo_name}-{env_name}.pk"
    with open(filename, "wb") as f:
        pickle.dump(training_rewards, f)


def training_evaluate(algo, env_name, interval, n=15):
    params = {
        "env_name": env_name,
        "actor_hidden_size": [128, 128],
        "critic_hidden_size": [128, 128],
        "noise_std": 0.1,
        "replay_size": 100_000,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "tau": 0.01,
        "gamma": 0.99,
        "batch_size": 128,
        "max_episodes": 500,
        "max_steps": 500,
        "policy_hidden_size": [128, 128],
        "q_hidden_size": [128, 128],
        "policy_lr": 3e-4,
        "q_lr": 3e-4,
        "alpha_lr": 3e-4,
        "hidden_size": [128, 128],
        "log_std_min": -20.0,
        "log_std_max": 2.0,
        "gradient_reduction": 0.8,
        "learning_rate": 3e-4,
    }

    training_rewards = {}

    for i in range(n):
        training_steps = 0

        episode_rewards = []
        agent = algo(params)
        progress = tqdm(range(params["max_episodes"]), desc=f"Training {i}/{n}")

        for episode in progress:
            state, _ = agent.train_env.reset()
            episode_reward = 0.0

            for step in range(agent.params["max_steps"]):
                action = agent.select_action(state)
                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = agent.train_env.step(action)
                done = terminated or truncated

                agent.replay_buffer.push(state, action, reward, next_state, done)
                agent.update()

                state = next_state
                episode_reward += float(reward)
                agent.total_steps += 1
                training_steps += 1

                if training_steps % interval == 0:
                    results = evaluation(agent, env_name)
                    id = training_steps / interval
                    if id not in training_rewards:
                        training_rewards[id] = []
                    training_rewards[id].extend(results)

                if done:
                    break

            episode_rewards.append(episode_reward)

            if len(episode_rewards) >= 10:
                avg_reward = np.mean(episode_rewards[-10:])
                progress.set_postfix(
                    {"avg_reward": f"{avg_reward:.2f}"}, refresh=True
                )

    Path("results").mkdir(exist_ok=True)

    algo_name = algo.__name__
    filename = f"results/{algo_name}-{env_name}.pk"
    with open(filename, "wb") as f:
        pickle.dump(training_rewards, f)


def main() -> None:
    algorithms = [DDPG, SAC, AFU]
    # training_evaluate(AFU, "CartPoleContinuousStudy-v0", interval=100, n=15)
    off_policy_training_evaluate(
        AFU,
        "CartPoleContinuousStudy-v0",
        total_steps=50_000,
        interval=100,
        n=15,
    )


if __name__ == "__main__":
    main()
