"""Tests for the Step 3 WFQ environment."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from wfq.env import WFQEnv


# Load project configuration
ROOT = Path(__file__).resolve().parents[1]

CONFIG = yaml.safe_load(
    (ROOT / "config" / "base.yaml").read_text(encoding="utf-8")
)

SIM = CONFIG["simulator"]

RATES = SIM["arrival_rates"]
CAPACITY = SIM["queue_capacity"]
SCV = SIM["scv"]
BANDWIDTH = SIM["total_bandwidth"]

# Read exact keys from base.yaml
DECISION_INTERVAL = SIM["decision_interval_s"]
EPISODE_LENGTH = SIM["episode_length_s"]

N = SIM["n_queues"]
EQUAL = [1.0] * N


def make_env():
    """Create a fresh WFQ environment."""

    return WFQEnv(
        arrival_rates=RATES,
        capacity=CAPACITY,
        scv=SCV,
        bandwidth=BANDWIDTH,
        decision_interval_seconds=DECISION_INTERVAL,
        episode_length_seconds=EPISODE_LENGTH,
    )


def summary(env):
    """Return packet statistics for every queue."""

    return [
        (q.arrived, q.served, q.lost)
        for q in env.system.queues
    ]


def test_weights_are_normalized_and_scale_does_not_change_behavior():

    first = make_env()
    first.reset(seed=21)

    state_a, reward_a, done_a, info_a = first.step([2.0] * N)

    second = make_env()
    second.reset(seed=21)

    state_b, reward_b, done_b, info_b = second.step([1.0] * N)

    expected = [1.0 / N] * N

    assert info_a["normalized_weights"] == pytest.approx(expected)
    assert info_b["normalized_weights"] == pytest.approx(expected)

    assert np.array_equal(state_a, state_b)
    assert reward_a == reward_b
    assert done_a == done_b

    assert summary(first) == summary(second)


@pytest.mark.parametrize(
    "bad_weights",
    [
        [0.0] + [1.0] * (N - 1),
        [-1.0] + [1.0] * (N - 1),
    ],
)
def test_zero_or_negative_weight_is_refused(bad_weights):

    env = make_env()
    env.reset(seed=22)

    with pytest.raises(ValueError):
        env.step(bad_weights)


def test_state_has_one_normalized_value_per_queue():

    env = make_env()

    state = env.reset(seed=23)

    assert len(state) == N
    assert np.all(state >= 0.0)
    assert np.all(state <= 1.0)

    next_state, _, _, _ = env.step(EQUAL)

    assert len(next_state) == N
    assert np.all(next_state >= 0.0)
    assert np.all(next_state <= 1.0)


def test_episode_has_100_decisions_and_done_remains_false():

    env = make_env()
    env.reset(seed=24)

    assert env.steps_per_episode == (
        EPISODE_LENGTH // DECISION_INTERVAL
    )

    assert env.steps_per_episode == 100

    for step in range(1, env.steps_per_episode + 1):

        state, reward, done, info = env.step(EQUAL)

        assert done is False
        assert env.step_count == step
        assert len(state) == N

    assert env.step_count == 100


def test_reset_reproducibility_and_different_seeds():

    # First run with seed 25
    first = make_env()

    initial_a = first.reset(seed=25)

    state_a, reward_a, _, _ = first.step(EQUAL)

    # Second run with the same seed
    second = make_env()

    initial_b = second.reset(seed=25)

    state_b, reward_b, _, _ = second.step(EQUAL)

    # Same seed must produce identical results
    assert np.array_equal(initial_a, initial_b)
    assert np.array_equal(state_a, state_b)
    assert reward_a == reward_b

    # Different seed
    third = make_env()

    third.reset(seed=26)

    state_c, reward_c, _, _ = third.step(EQUAL)

    assert (
        not np.array_equal(state_a, state_c)
        or reward_a != reward_c
    )


def test_packet_balance_holds_for_every_queue_over_an_episode():

    env = make_env()
    env.reset(seed=27)

    for _ in range(env.steps_per_episode):
        env.step(EQUAL)

    for q in env.system.queues:

        assert q.arrived == (
            q.served + q.lost + q.in_system()
        )


def test_overloaded_queues_lose_packets_but_light_queue_does_not():

    env = make_env()
    env.reset(seed=2)

    for _ in range(env.steps_per_episode):
        env.step(EQUAL)

    # Queues 2, 3 and 6 are overloaded
    for index in (1, 2, 5):
        assert env.system.queues[index].lost > 0

    # Queue 1 is lightly loaded
    assert env.system.queues[0].lost == 0