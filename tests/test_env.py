"""Tests for the WFQ environment (Step 3 build, Step 4 checks).

Every number comes from config/base.yaml, read with its exact key name.
If a key is renamed or missing, these tests fail with a KeyError
instead of quietly using a default value.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from wfq.env import WFQEnv


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
SIM = CONFIG["simulator"]

RATES = SIM["arrival_rates"]
N = SIM["n_queues"]
CAPACITY = SIM["queue_capacity"]
SCV = SIM["scv"]
BANDWIDTH = SIM["total_bandwidth"]
DECISION_INTERVAL = SIM["decision_interval_s"]
EPISODE_LENGTH = SIM["episode_length_s"]

EQUAL = [1.0] * N
EXPECTED_STEPS = round(EPISODE_LENGTH / DECISION_INTERVAL)


def make_env():
    return WFQEnv(
        arrival_rates=RATES,
        capacity=CAPACITY,
        scv=SCV,
        bandwidth=BANDWIDTH,
        decision_interval_seconds=DECISION_INTERVAL,
        episode_length_seconds=EPISODE_LENGTH,
    )


def summary(env):
    return [(q.arrived, q.served, q.lost) for q in env.system.queues]


def test_env_reads_the_settings_file():
    env = make_env()

    assert len(RATES) == N
    assert env.n_queues == N
    assert env.decision_interval_seconds == DECISION_INTERVAL
    assert env.episode_length_seconds == EPISODE_LENGTH
    assert env.steps_per_episode == EXPECTED_STEPS


def test_weights_are_normalized_and_scale_does_not_change_behavior():
    first = make_env()
    first.reset(seed=21)
    state_a, reward_a, _, info_a = first.step([2.0] * N)

    second = make_env()
    second.reset(seed=21)
    state_b, reward_b, _, info_b = second.step([1.0] * N)

    assert info_a["normalized_weights"] == pytest.approx([1.0 / N] * N)
    assert state_a.tolist() == state_b.tolist()
    assert reward_a == reward_b
    assert summary(first) == summary(second)


def test_uneven_weights_are_scaled_to_sum_to_one():
    env = make_env()
    env.reset(seed=20)

    uneven = [float(i + 1) for i in range(N)]
    total = sum(uneven)

    _, _, _, info = env.step(uneven)
    normalized = np.array(info["normalized_weights"])

    assert normalized.sum() == pytest.approx(1.0)
    assert normalized.tolist() == pytest.approx([w / total for w in uneven])
    assert np.all(normalized > 0)


@pytest.mark.parametrize("bad_weights", [
    [0.0] + [1.0] * (N - 1),
    [-1.0] + [1.0] * (N - 1),
])
def test_zero_or_negative_weight_is_refused(bad_weights):
    env = make_env()
    env.reset(seed=22)

    with pytest.raises(ValueError):
        env.step(bad_weights)


def test_wrong_number_of_weights_is_refused():
    env = make_env()
    env.reset(seed=28)

    with pytest.raises(ValueError):
        env.step([1.0] * (N - 1))


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


def test_episode_runs_for_exact_steps_and_done_stays_false():
    env = make_env()
    env.reset(seed=24)

    assert env.steps_per_episode == EXPECTED_STEPS

    for step in range(1, env.steps_per_episode + 1):
        _, _, done, info = env.step(EQUAL)

        assert done is False
        assert info["step"] == step

    assert env.step_count == env.steps_per_episode


def test_reset_reproducibility_and_different_seeds():
    first = make_env()
    initial_a = first.reset(seed=25)
    state_a, reward_a, _, _ = first.step(EQUAL)

    second = make_env()
    initial_b = second.reset(seed=25)
    state_b, reward_b, _, _ = second.step(EQUAL)

    assert np.array_equal(initial_a, initial_b)
    assert np.array_equal(state_a, state_b)
    assert reward_a == reward_b

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
        assert q.arrived == q.served + q.lost + q.in_system()


def test_overloaded_queues_lose_packets_but_light_queue_does_not():
    env = make_env()
    env.reset(seed=2)

    for _ in range(env.steps_per_episode):
        env.step(EQUAL)

    for index in (1, 2, 5):
        assert env.system.queues[index].lost > 0

    assert env.system.queues[0].lost == 0
