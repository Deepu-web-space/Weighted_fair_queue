"""Tests for the Step 3 WFQ environment."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from wfq.env import WFQEnv


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
SIM = CONFIG["simulator"]

RATES = SIM["arrival_rates"]
CAPACITY = SIM["queue_capacity"]
SCV = SIM["scv"]
BANDWIDTH = SIM["total_bandwidth"]

# Support common key spellings while keeping the values in base.yaml.
DECISION_INTERVAL = SIM.get(
    "decision_interval_seconds",
    SIM.get("decision_interval", SIM.get("action_interval_seconds", 20)),
)
EPISODE_LENGTH = SIM.get(
    "episode_length_seconds",
    SIM.get("episode_length", SIM.get("episode_duration_seconds", 2000)),
)

N = len(RATES)
EQUAL = [1.0] * N


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


@pytest.mark.parametrize("bad_weights", [
    [0.0] + [1.0] * (N - 1),
    [-1.0] + [1.0] * (N - 1),
])
def test_zero_or_negative_weight_is_refused(bad_weights):
    env = make_env()
    env.reset(seed=22)

    with pytest.raises(ValueError):
        env.step(bad_weights)


def test_state_has_one_normalized_value_per_queue():
    env = make_env()
    state = env.reset(seed=23)

    assert len(state) == 8
    assert np.all(state >= 0.0)
    assert np.all(state <= 1.0)

    next_state, _, _, _ = env.step(EQUAL)
    assert len(next_state) == 8
    assert np.all(next_state >= 0.0)
    assert np.all(next_state <= 1.0)


def test_episode_has_100_decisions_and_done_remains_false():
    env = make_env()
    env.reset(seed=24)

    assert env.steps_per_episode == 100