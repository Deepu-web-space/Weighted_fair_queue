"""Tests for Day 3, Block 2: the replay buffer."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from wfq.Replay import ReplayBuffer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
SIM = CONFIG["simulator"]
TRAIN = CONFIG["training"]
N = SIM["n_queues"]


def _fill(buf, count):
    for i in range(count):
        s = np.full(N, float(i), dtype=np.float32)
        buf.add(s, s, float(i), s, False)


def test_capacity_comes_from_settings():
    buf = ReplayBuffer(TRAIN["replay_size"], N, N)
    assert buf.capacity == TRAIN["replay_size"]
    assert buf.states.shape == (TRAIN["replay_size"], N)


def test_size_grows_then_stops_at_capacity():
    buf = ReplayBuffer(5, N, N)
    _fill(buf, 3)
    assert buf.size == 3
    _fill(buf, 10)
    assert buf.size == 5


def test_oldest_transition_is_overwritten_first():
    buf = ReplayBuffer(4, N, N)
    _fill(buf, 6)  # values 0..5; slots now hold 4, 5, 2, 3
    assert buf.rewards[0, 0] == 4.0
    assert buf.rewards[1, 0] == 5.0
    assert buf.rewards[2, 0] == 2.0
    assert buf.rewards[3, 0] == 3.0


def test_sample_shapes():
    buf = ReplayBuffer(TRAIN["replay_size"], N, N)
    _fill(buf, 50)
    s, a, r, s2, d = buf.sample(8, np.random.default_rng(0))
    assert s.shape == (8, N)
    assert a.shape == (8, N)
    assert r.shape == (8, 1)
    assert s2.shape == (8, N)
    assert d.shape == (8, 1)


def test_sample_is_reproducible_with_same_seed():
    buf = ReplayBuffer(TRAIN["replay_size"], N, N)
    _fill(buf, 50)
    first = buf.sample(8, np.random.default_rng(7))[2]
    second = buf.sample(8, np.random.default_rng(7))[2]
    assert np.array_equal(first, second)


def test_sample_refuses_more_than_stored():
    buf = ReplayBuffer(TRAIN["replay_size"], N, N)
    _fill(buf, 3)
    with pytest.raises(ValueError):
        buf.sample(4, np.random.default_rng(0))


def test_zero_capacity_refused():
    with pytest.raises(ValueError):
        ReplayBuffer(0, N, N)
