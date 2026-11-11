
"""WFQ environment wrapper for reinforcement learning."""

import math
import numpy as np

from wfq.simulator import QueueSystem


class WFQEnv:
    """Wrap QueueSystem with fixed-duration agent decision steps."""

    def __init__(
        self,
        arrival_rates,
        capacity,
        scv,
        bandwidth,
        decision_interval_seconds,
        episode_length_seconds,
    ):

        if len(arrival_rates) == 0:
            raise ValueError("arrival_rates must contain at least one queue")

        if capacity < 1:
            raise ValueError("capacity must be at least 1")

        if scv <= 0:
            raise ValueError("scv must be positive")

        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")

        if decision_interval_seconds <= 0:
            raise ValueError("decision interval must be positive")

        if episode_length_seconds <= 0:
            raise ValueError("episode length must be positive")

        steps = episode_length_seconds / decision_interval_seconds

        if not math.isclose(steps, round(steps), abs_tol=1e-10):
            raise ValueError(
                "Episode length must be an exact multiple "
                "of decision interval"
            )

        self.arrival_rates = list(arrival_rates)
        self.capacity = capacity
        self.scv = scv
        self.bandwidth = bandwidth

        self.decision_interval_seconds = float(
            decision_interval_seconds
        )

        self.episode_length_seconds = float(
            episode_length_seconds
        )

        self.n_queues = len(self.arrival_rates)

        self.steps_per_episode = int(round(steps))

        self.system = None
        self.step_count = 0

    def _state(self):
        """Return normalized queue lengths."""

        return np.asarray(
            [
                length / self.capacity
                for length in self.system.lengths()
            ],
            dtype=np.float32,
        )

    def reset(self, seed):
        """Initialize a fresh simulation."""

        equal_weights = [1.0 / self.n_queues] * self.n_queues

        self.system = QueueSystem(
            self.arrival_rates,
            equal_weights,
            self.capacity,
            self.scv,
            self.bandwidth,
            seed=seed,
        )

        self.step_count = 0

        return self._state()

    def step(self, weights):
        """Apply weights and advance the simulation."""

        if self.system is None:
            raise RuntimeError("Call reset(seed) before step(weights)")

        weights = np.asarray(weights, dtype=float)

        # Validate action
        if weights.ndim != 1 or len(weights) != self.n_queues:
            raise ValueError("One weight is required per queue")

        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("Weights must be finite and positive")

        # Normalize weights
        normalized_weights = weights / np.sum(weights)

        # Record counters before the step
        lost_before = [
            q.lost for q in self.system.queues
        ]

        served_before = [
            q.served for q in self.system.queues
        ]

        # Apply weights
        self.system.set_weights(
            normalized_weights.tolist()
        )

        # Advance simulation
        self.system.advance(
            self.decision_interval_seconds
        )

        # Calculate per-step losses
        lost_counts = [
            q.lost - before
            for q, before in zip(
                self.system.queues,
                lost_before
            )
        ]

        # Calculate per-step served packets
        served_counts = [
            q.served - before
            for q, before in zip(
                self.system.queues,
                served_before
            )
        ]

        # Get current state
        state = self._state()

        queue_lengths = [
            q.in_system()
            for q in self.system.queues
        ]

        # Calculate reward components
        holding_cost = (
            sum(queue_lengths) / self.n_queues
        )

        loss_penalty = sum(lost_counts)

        # Paper reward equation
        reward = -(holding_cost + loss_penalty)

        # Update step counter
        self.step_count += 1

        # Information dictionary
        info = {
            "served_counts": served_counts,
            "lost_counts": lost_counts,
            "total_lost": int(sum(lost_counts)),
            "holding_cost": float(holding_cost),
            "loss_penalty": float(loss_penalty),
            "normalized_weights": normalized_weights.tolist(),
            "queue_lengths": queue_lengths,
            "step": self.step_count,
        }

        # Episode truncation is handled by the training loop.
        done = False

        # IMPORTANT: Return all four values.
        return state, float(reward), done, info
