"""Tests for Day 3, Block 2: the actor and critic networks.

Settings are read from config/base.yaml. Nothing is typed in except
the shape checks.
"""

from pathlib import Path

import torch
import yaml

from wfq.Networks import Actor, Critic


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
SIM = CONFIG["simulator"]
TRAIN = CONFIG["training"]
N = SIM["n_queues"]


def _random_states(batch):
    # States are queue fill levels between 0 and 1.
    return torch.rand(batch, N)


def test_actor_weights_are_positive_and_sum_to_one():
    torch.manual_seed(0)
    actor = Actor(N, TRAIN["hidden_units"], TRAIN["min_concentration"])
    w = actor(_random_states(16)).sample()

    assert w.shape == (16, N)
    assert torch.all(w >= 0)
    assert torch.allclose(w.sum(dim=-1), torch.ones(16), atol=1e-5)


def test_actor_mean_weights_sum_to_one_and_are_positive():
    torch.manual_seed(0)
    actor = Actor(N, TRAIN["hidden_units"], TRAIN["min_concentration"])
    w = actor.mean_weights(_random_states(5))

    assert w.shape == (5, N)
    assert torch.all(w > 0)
    assert torch.allclose(w.sum(dim=-1), torch.ones(5), atol=1e-6)


def test_concentration_stays_above_minimum():
    torch.manual_seed(0)
    actor = Actor(N, TRAIN["hidden_units"], TRAIN["min_concentration"])
    c = actor.concentration(_random_states(32))

    assert torch.all(c > TRAIN["min_concentration"])


def test_actor_log_prob_is_finite_for_samples():
    torch.manual_seed(0)
    actor = Actor(N, TRAIN["hidden_units"], TRAIN["min_concentration"])
    s = _random_states(8)
    dist = actor(s)
    w = dist.rsample()
    logp = dist.log_prob(w)

    assert logp.shape == (8,)
    assert torch.all(torch.isfinite(logp))


def test_actor_hidden_layers_use_hidden_units_setting():
    actor = Actor(N, TRAIN["hidden_units"], TRAIN["min_concentration"])
    first = actor.net[0]
    assert first.in_features == N
    assert first.out_features == TRAIN["hidden_units"]


def test_critic_outputs_one_value_per_row():
    torch.manual_seed(0)
    critic = Critic(N, TRAIN["hidden_units"])
    s = _random_states(7)
    w = torch.full((7, N), 1.0 / N)
    q = critic(s, w)

    assert q.shape == (7, 1)
    assert torch.all(torch.isfinite(q))
