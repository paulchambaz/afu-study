import numpy as np
import torch
from collections import deque
import random


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

    def sample(self, batch_size: int, continuous: bool):
        """Randomly sample a batch of transitions for training."""

        # Take a random sample of transitions from our buffer to break correlations
        # in sequential data.
        transitions = random.sample(self.buffer, batch_size)

        # Reorganize the batch of transitions into separate arrays for each component.
        # This transforms a list of (state, action, reward, next_state, done) tuples
        # into separate arrays for states, actions, rewards, etc.
        batch = list(zip(*transitions))

        # Convert to numpy arrays first to ensure consistent shapes
        _states = np.array(batch[0])
        _actions = np.array(batch[1])
        _rewards = np.array(batch[2])
        _next_states = np.array(batch[3])
        _dones = np.array(batch[4])

        # Convert the arrays into PyTorch tensors with appropriate types for training.
        # States and next_states are converted to floating point tensors.
        # Actions are converted to long tensors for discrete action spaces.
        # Rewards and done flags are converted to floating point tensors.
        states = torch.FloatTensor(_states)
        actions = (
            torch.FloatTensor(_actions.reshape(batch_size, -1))
            if continuous
            else torch.LongTensor(_actions)
        )
        rewards = torch.FloatTensor(_rewards)
        next_states = torch.FloatTensor(_next_states)
        dones = torch.FloatTensor(_dones)

        return states, actions, rewards, next_states, dones

    def get_latest(self):
        """Retrieve the most recently added transition."""

        latest = self.buffer[-1]

        state = np.array(latest[0])
        action = np.array(latest[1])
        reward = np.array(latest[2])
        next_state = np.array(latest[3])
        done = np.array(latest[4])

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
