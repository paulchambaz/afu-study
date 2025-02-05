import torch
import gymnasium as gym
import torch.nn as nn
import random
import numpy as np
from bbrl.agents import Agent # type: ignore
from bbrl_utils.nn import build_mlp # type: ignore
from bbrl.workspace import Workspace # type: ignore
from collections import deque

class DiscreteQNetwork(Agent):
    def __init__(self, state_dim, hidden_size, action_dim):
        super().__init__()
        self.model = build_mlp(
            [state_dim] + hidden_size + [action_dim],
            activation=nn.ReLU()
        )

    def forward(self, t):
        obs = self.get(("env/env_obs", t))
        q_values = self.model(obs)
        self.set(("q_values", t), q_values)

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        batch = list(zip(*transitions))

        states = torch.FloatTensor(np.array(batch[0]))
        actions = torch.LongTensor(batch[1])
        rewards = torch.FloatTensor(batch[2])
        next_states = torch.FloatTensor(np.array(batch[3]))
        dones = torch.FloatTensor(batch[4])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DQN:
    def __init__(self, params):
        self.params = params

        self.train_env = gym.make(params["env_name"])

        state_dim = self.train_env.observation_space.shape[0]
        action_dim = self.train_env.action_space.n

        self.q_network = DiscreteQNetwork(state_dim, params["hidden_size"], action_dim)
        self.target_network = DiscreteQNetwork(state_dim, params["hidden_size"], action_dim)

        self.target_network.load_state_dict(self.q_network.state_dict())

        self.replay_buffer = ReplayBuffer(params["replay_size"])

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=params["learning_rate"]
        )

        self.total_steps = 0

    def select_action(self, state, evaluation=False):
        if evaluation:
            epsilon = 0.0
        else:
            epsilon = self.params["epsilon_end"] + (self.params["epsilon_start"] - self.params["epsilon_end"]) * np.exp(-self.total_steps / self.params["epsilon_decay"])

        workspace = Workspace()
        workspace.set("env/env_obs", 0, torch.FloatTensor([state]))

        self.q_network(workspace, t=0)
        q_values = workspace.get("q_values", 0)

        if random.random() > epsilon:
            action = q_values.argmax().item()
        else:
            action = self.train_env.action_space.sample()

        return action

    def update(self):
        if len(self.replay_buffer) < self.params["batch_size"]:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.params["batch_size"])

        workspace = Workspace()
        workspace.set("env/env_obs", 0, states)
        self.q_network(workspace, t=0)
        q_values = workspace.get("q_values", 0)

        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        workspace.set("env/env_obs", 0, next_states)
        self.target_network(workspace, t=0)
        next_q_values = workspace.get("q_values", 0)

        targets = rewards + (1 - dones) * self.params["gamma"] * next_q_values.max(1)[0].detach()

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_steps % self.params["target_update"] == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def train(self):
        episode_rewards = []

        for episode in range(self.params["max_episodes"]):
            state, _ = self.train_env.reset()

            episode_reward = 0

            for step in range(self.params["max_steps"]):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.train_env.step(action)
                done = terminated or truncated

                self.replay_buffer.push(state, action, reward, next_state, done)

                _ = self.update()

                state = next_state
                episode_reward += reward
                self.total_steps += 1

                if done:
                    break

            episode_rewards.append(episode_reward)

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")

        return {"episode_rewards": episode_rewards}

    def save(self, path):
        save_dict = {
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'params': self.params,
            'total_steps': self.total_steps
        }

        torch.save(save_dict, path)
        print(f"Agent saved successfully to {path}")

    def load(self, path):
        save_dict = torch.load(path)
        
        self.q_network.load_state_dict(save_dict['q_network_state'])
        self.target_network.load_state_dict(save_dict['target_network_state'])
        self.optimizer.load_state_dict(save_dict['optimizer_state'])
        
        self.params = save_dict['params']
        self.total_steps = save_dict['total_steps']
        
        print(f"Agent loaded successfully from {path}")

    @classmethod
    def load_agent(cls, path):
        save_dict = torch.load(path)
        agent = cls(save_dict['params'])
        agent.load(path)
        return agent
