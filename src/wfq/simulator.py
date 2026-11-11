"""Step 1: the packet queues.

Each traffic flow has its own waiting line with one server.
A packet that arrives when its line is full is lost.
Time moves forward from one event to the next: a packet arrives,
or the packet being served finishes.

No numbers are typed in this file. The caller passes every value in,
and the caller gets the values from config/base.yaml.
"""

import math

import numpy as np


class PacketQueue:
    """One waiting line with a single server."""

    def __init__(self, arrival_rate, weight, capacity, scv, bandwidth, rng):
        if arrival_rate <= 0:
            raise ValueError("arrival_rate must be positive")
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if scv <= 0:
            raise ValueError("scv must be positive")

        self.arrival_rate = arrival_rate
        self.capacity = capacity
        self.scv = scv
        self.bandwidth = bandwidth
        self.rng = rng

        self.weight = None
        self.set_weight(weight)

        self.now = 0.0
        self.waiting = 0                          # packets in line, not being served
        self.busy = False                         # True while one packet is being served
        self.service_done_at = math.inf           # when the packet in service finishes
        self.arrived = 0                          # every packet that tried to join
        self.served = 0                           # packets that finished service
        self.lost = 0                             # packets turned away because the line was full

        self.next_arrival = self._draw_gap()

    def set_weight(self, weight):
        """Change the weight. It applies to the next packet that starts service."""
        if weight <= 0:
            raise ValueError("weight must be positive")
        self.weight = weight

    def in_system(self):
        """Packets in the line plus the one being served."""
        return self.waiting + (1 if self.busy else 0)

    def _gamma(self, mean):
        # Gamma with squared coefficient of variation = scv:
        # shape = 1 / scv, scale = scv * mean. The mean stays equal to `mean`.
        shape = 1.0 / self.scv
        scale = self.scv * mean
        return self.rng.gamma(shape, scale)

    def _draw_gap(self):
        """Time until the next packet arrives."""
        return self._gamma(1.0 / self.arrival_rate)

    def _draw_service(self):
        """Time to serve one packet. Service speed = weight x bandwidth."""
        service_rate = self.weight * self.bandwidth
        return self._gamma(1.0 / service_rate)

    def _start_service(self, at_time):
        self.busy = True
        self.service_done_at = at_time + self._draw_service()

    def advance(self, until):
        """Run this queue forward until time `until`."""
        if until < self.now:
            raise ValueError("cannot move back in time")

        while True:
            if self.next_arrival <= self.service_done_at and self.next_arrival <= until:
                self._arrive(self.next_arrival)
            elif self.service_done_at <= until:
                self._finish(self.service_done_at)
            else:
                break

        self.now = until

    def _arrive(self, t):
        self.arrived += 1
        if self.in_system() < self.capacity:
            if self.busy:
                self.waiting += 1
            else:
                self._start_service(t)
        else:
            self.lost += 1
        self.next_arrival = t + self._draw_gap()

    def _finish(self, t):
        self.served += 1
        self.busy = False
        self.service_done_at = math.inf
        if self.waiting > 0:
            self.waiting -= 1
            self._start_service(t)


class QueueSystem:
    """All the flows together. They share one random generator, so one seed controls everything."""

    def __init__(self, arrival_rates, weights, capacity, scv, bandwidth, seed):
        if len(arrival_rates) != len(weights):
            raise ValueError("need exactly one weight per queue")

        self.rng = np.random.default_rng(seed)
        self.queues = [
            PacketQueue(rate, weight, capacity, scv, bandwidth, self.rng)
            for rate, weight in zip(arrival_rates, weights)
        ]
        self.now = 0.0

    def set_weights(self, weights):
        for queue, weight in zip(self.queues, weights):
            queue.set_weight(weight)

    def advance(self, seconds):
        """Run every queue forward by `seconds`."""
        self.now += seconds
        for queue in self.queues:
            queue.advance(self.now)

    def lengths(self):
        """Packets currently in each queue."""
        return [queue.in_system() for queue in self.queues]
