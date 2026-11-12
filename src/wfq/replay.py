"""Replay buffer for SAC (Day 3, Block 2).

Stores past transitions (state, weights, reward, next state, done) and
returns random batches for learning. When full, the oldest transition is
overwritten first.
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")

        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        i = self.position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        """Return a random batch. rng is a numpy Generator for reproducibility."""
        if batch_size > self.size:
            raise ValueError(
                f"only {self.size} transitions stored, "
                f"cannot sample {batch_size}"
            )

        idx = rng.integers(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )
