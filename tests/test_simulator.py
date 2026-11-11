"""Tests for Step 1: the packet queues.

The values come from config/base.yaml, so the tests and the experiments
always use the same numbers.
"""

from pathlib import Path

import pytest
import yaml

from wfq.simulator import QueueSystem

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config" / "base.yaml").read_text())
SIM = CONFIG["simulator"]

RATES = SIM["arrival_rates"]
CAPACITY = SIM["queue_capacity"]
SCV = SIM["scv"]
BANDWIDTH = SIM["total_bandwidth"]
N = SIM["n_queues"]
EQUAL = [1.0 / N] * N


def make_system(seed):
    return QueueSystem(RATES, EQUAL, CAPACITY, SCV, BANDWIDTH, seed=seed)


def test_packet_balance_holds_for_every_queue():
    # Every packet that arrived is either served, lost, or still in the queue.
    system = make_system(seed=1)
    for _ in range(100):  # 100 decisions of 20 s = 2000 s
        system.advance(20)
        for q in system.queues:
            assert q.arrived == q.served + q.lost + q.in_system()


def test_no_queue_ever_holds_more_than_capacity():
    system = make_system(seed=2)
    worst = 0
    for _ in range(2000):
        system.advance(1)
        worst = max(worst, max(system.lengths()))
    assert worst <= CAPACITY


def test_overloaded_queues_lose_packets_and_light_queue_does_not():
    # Under equal weights each queue serves 0.125 packets/s.
    # Queues 2, 3 and 6 (indexes 1, 2, 5) get more than that, so they must lose packets.
    # Queue 1 (index 0) gets 0.03 packets/s, so it should lose none in 2000 s.
    system = make_system(seed=2)
    system.advance(2000)
    for index in (1, 2, 5):
        assert system.queues[index].lost > 0
    assert system.queues[0].lost == 0


def test_arrival_rate_matches_setting():
    # The arrivals do not depend on the service, so a long run gives a clean check.
    seconds = 100_000
    system = make_system(seed=3)
    system.advance(seconds)
    for q, rate in zip(system.queues, RATES):
        measured = q.arrived / seconds
        assert measured == pytest.approx(rate, rel=0.05)


def test_same_seed_gives_same_numbers():
    first = make_system(seed=4)
    first.advance(2000)
    second = make_system(seed=4)
    second.advance(2000)
    summary_first = [(q.arrived, q.served, q.lost) for q in first.queues]
    summary_second = [(q.arrived, q.served, q.lost) for q in second.queues]
    assert summary_first == summary_second


def test_zero_weight_is_refused():
    system = make_system(seed=5)
    with pytest.raises(ValueError):
        system.set_weights([0.0] + EQUAL[1:])
