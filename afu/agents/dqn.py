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
    """A neural network that maps states to Q-values for discrete actions."""

    def __init__(self, state_dim, hidden_size, action_dim):
        """Initialize Q-network with given dimensions."""
        super().__init__()

        # Build a neural network that maps state observations to Q-values. The network 
        # takes state_dim inputs and outputs action_dim Q-values, with hidden_size
        # determining the size of intermediate layers. ReLU activations introduce 
        # non-linearity while maintaining good gradient flow.
        self.model = build_mlp(
            [state_dim] + hidden_size + [action_dim],
            activation=nn.ReLU()
        )

    def forward(self, t):
        """Compute Q-values for a given state observation."""

        # Get the current state observation from the workspace at time t. The workspace
        # stores environment observations that can be accessed by any agent.
        obs = self.get(("env/env_obs", t))

        # Pass the observation through our neural network to get Q-values for each
        # possible action in the current state.
        q_values = self.model(obs)

        # Store the computed Q-values back in the workspace so they can be used
        # by other agents for action selection or learning.
        self.set(("q_values", t), q_values)

class ReplayBuffer:
    """Store and sample transitions for experience replay in reinforcement learning."""

    def __init__(self, max_size):
        """Initialize buffer with maximum capacity."""

        # Create a double-ended queue with fixed maximum size. When the buffer is full,
        # adding new items automatically removes the oldest ones.
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        """Add a new transition to the replay buffer."""

        # Store a complete transition as a tuple containing all information needed
        # for learning: current state, action taken, reward received, next state,
        # and whether the episode ended.
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly sample a batch of transitions for training."""

        # Take a random sample of transitions from our buffer to break correlations
        # in sequential data.
        transitions = random.sample(self.buffer, batch_size)

        # Reorganize the batch of transitions into separate arrays for each component.
        # This transforms a list of (state, action, reward, next_state, done) tuples
        # into separate arrays for states, actions, rewards, etc.
        batch = list(zip(*transitions))

        # Convert the arrays into PyTorch tensors with appropriate types for training.
        # States and next_states are converted to floating point tensors.
        # Actions are converted to long tensors for discrete action spaces.
        # Rewards and done flags are converted to floating point tensors.
        states = torch.FloatTensor(np.array(batch[0]))
        actions = torch.LongTensor(batch[1])
        rewards = torch.FloatTensor(batch[2])
        next_states = torch.FloatTensor(np.array(batch[3]))
        dones = torch.FloatTensor(batch[4])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DQN:
    """Deep Q-Network agent that learns optimal actions through experience replay and target networks."""

    def __init__(self, params):
        """Initialize DQN agent with training parameters."""

        self.params = params

        # Create the training environment and get its dimensions. The observation_space
        # represents states (like positions and velocities in CartPole) while 
        # action_space.n gives us the number of possible discrete actions.
        self.train_env = gym.make(params["env_name"])
        state_dim = self.train_env.observation_space.shape[0]
        action_dim = self.train_env.action_space.n

        # Create two networks: the main Q-network for training and a target network
        # for stable Q-value estimates. The target network parameters are only
        # updated periodically to prevent unstable learning from a moving target.
        self.q_network = DiscreteQNetwork(state_dim, params["hidden_size"], action_dim)
        self.target_network = DiscreteQNetwork(state_dim, params["hidden_size"], action_dim)

        # Initialize target network with same weights as Q-network. state_dict()
        # contains all the network parameters (weights and biases) that we copy over.
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Create replay buffer for experience replay, which breaks correlations
        # in the training data and allows multiple updates from each transition.
        self.replay_buffer = ReplayBuffer(params["replay_size"])

        # Setup Adam optimizer which adapts learning rates for each parameter
        # and uses momentum to accelerate training while avoiding local optima.
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=params["learning_rate"]
        )

        self.total_steps = 0

    def select_action(self, state, evaluation=False):
        """Choose action using epsilon-greedy strategy to balance exploration and exploitation."""

        # During evaluation, we don't explore (epsilon = 0). During training, epsilon
        # decays exponentially from epsilon_start to epsilon_end, gradually reducing
        # random exploration as the agent learns.
        if evaluation:
            epsilon = 0.0
        else:
            epsilon = self.params["epsilon_end"] + (self.params["epsilon_start"] - self.params["epsilon_end"]) * np.exp(-self.total_steps / self.params["epsilon_decay"])

        # Use workspace to get Q-values from our network. This is BBRL's way of
        # passing data between agents - the workspace acts like a shared memory.
        workspace = Workspace()
        workspace.set("env/env_obs", 0, torch.FloatTensor([state]))
        self.q_network(workspace, t=0)
        q_values = workspace.get("q_values", 0)

        # Epsilon-greedy action selection: with probability epsilon, take random action
        # for exploration; otherwise choose action with highest Q-value for exploitation.
        if random.random() > epsilon:
            action = q_values.argmax().item()
        else:
            action = self.train_env.action_space.sample()

        return action

    def update(self):
        """Perform one step of Q-learning using a batch of experiences."""

        # Only update if we have enough transitions for a full batch
        if len(self.replay_buffer) < self.params["batch_size"]:
            return

        # Sample a batch of transitions from replay buffer for training
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.params["batch_size"])

        # Compute current Q-values using Q-network. We use workspace to interact
        # with our Q-network agent, following BBRL's agent communication pattern.
        workspace = Workspace()
        workspace.set("env/env_obs", 0, states)
        self.q_network(workspace, t=0)
        q_values = workspace.get("q_values", 0)

        # First, we need to get the Q-values that led to our chosen actions
        # q_values contains Q-values for all actions, but we only want the ones we took.
        # We are asking "what value did we expect from the actions we chose?"
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Now we need to calculate what values we SHOULD have expected
        # We do this using the target network which provides stable estimates
        workspace.set("env/env_obs", 0, next_states)
        self.target_network(workspace, t=0)
        next_q_values = workspace.get("q_values", 0)

        # This is the core of Q-learning: we compute the "true" value of our actions
        # If the episode didn't end (1 - dones), we add future rewards:
        #   - reward we got immediately
        #   - plus gamma (discount) times the best value possible from the next state
        # If the episode did end, we just get the immediate reward
        # detach() prevents us from trying to optimize the target network
        targets = rewards + (1 - dones) * self.params["gamma"] * next_q_values.max(1)[0].detach()

        # Compare our predictions (q_values) to what actually happened (targets)
        # This tells us how wrong our predictions were
        loss = nn.MSELoss()(q_values, targets)

        # Optimize: clear old gradients, compute new gradients, update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically copy Q-network weights to target network
        if self.total_steps % self.params["target_update"] == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def train(self):
        """Train the agent over multiple episodes."""

        # Track rewards for performance monitoring
        episode_rewards = []

        # Each episode is one complete run through the environment
        for episode in range(self.params["max_episodes"]):
            state, _ = self.train_env.reset()
            episode_reward = 0

            # Step through episode until done or max steps reached
            for step in range(self.params["max_steps"]):
                # Core loop: select action, execute it, store transition, update
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.train_env.step(action)
                done = terminated or truncated

                # Store transition and update networks
                self.replay_buffer.push(state, action, reward, next_state, done)
                _ = self.update()

                state = next_state
                episode_reward += reward
                self.total_steps += 1

                if done:
                    break

            # Store episode reward and periodically print progress
            episode_rewards.append(episode_reward)

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")

        return {"episode_rewards": episode_rewards}

    def save(self, path):
        """Save the complete state of the DQN agent to disk."""

        # Save everything needed to restore the agent: network weights,
        # optimizer state, parameters, and training progress
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
        """Load a previously saved DQN agent state."""

        # Load the saved dictionary and restore each component
        save_dict = torch.load(path)
        
        self.q_network.load_state_dict(save_dict['q_network_state'])
        self.target_network.load_state_dict(save_dict['target_network_state'])
        self.optimizer.load_state_dict(save_dict['optimizer_state'])
        
        self.params = save_dict['params']
        self.total_steps = save_dict['total_steps']
        
        print(f"Agent loaded successfully from {path}")

    @classmethod
    def load_agent(cls, path):
        """Create a new DQN agent from a saved file."""

        # First create a new agent, then load the saved state into it
        save_dict = torch.load(path)
        agent = cls(save_dict['params'])
        agent.load(path)
        return agent
