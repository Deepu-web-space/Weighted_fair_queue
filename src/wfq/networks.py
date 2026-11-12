"""Neural networks for SAC (Day 3, Block 2).

Actor: reads the fill level of each queue and outputs a Dirichlet
distribution over the weights. A Dirichlet sample is always positive and
always sums to 1, so it is a valid bandwidth split with no extra work.

Critic: reads a state and a weight vector and outputs one number, the
predicted total future reward (the Q value).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Maps a state to a Dirichlet distribution over the weights."""

    def __init__(self, n_queues, hidden_units, min_concentration):
        super().__init__()
        self.min_concentration = min_concentration
        self.net = nn.Sequential(
            nn.Linear(n_queues, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, n_queues),
        )

    def concentration(self, state):
        """Positive concentration values, one per queue.

        softplus keeps every value above zero. Adding min_concentration
        keeps the distribution from becoming very spiky.
        """
        return F.softplus(self.net(state)) + self.min_concentration

    def forward(self, state):
        """Return the Dirichlet distribution for this state."""
        # validate_args off: samples are floored and renormalised in the
        # agent, so float rounding must not trip the simplex check.
        return torch.distributions.Dirichlet(
            self.concentration(state), validate_args=False
        )

    def mean_weights(self, state):
        """Deterministic choice: the mean of the Dirichlet."""
        c = self.concentration(state)
        return c / c.sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    """Maps (state, weights) to one predicted value."""

    def __init__(self, n_queues, hidden_units):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * n_queues, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, state, weights):
        return self.net(torch.cat([state, weights], dim=-1))
