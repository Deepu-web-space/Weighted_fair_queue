"""Tests for Day 3, Block 2: the SAC agent without EWC.

Settings come from config/base.yaml. The last test runs the agent
against the real environment for two short episodes.
"""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from wfq.env import WFQEnv
from wfq.Sac import SACAgent


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
SIM = CONFIG["simulator"]
TRAIN = CONFIG["training"]
N = SIM["n_queues"]


def _random_transition(rng):
    s = rng.random(N, dtype=np.float32)
    w = rng.dirichlet(np.ones(N)).astype(np.float32)
    r = float(-rng.random() * 10)
    s2 = rng.random(N, dtype=np.float32)
    return s, w, r, s2, False


def _fill_agent(agent, count, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        agent.store(*_random_transition(rng))


def _params(module):
    return [p.detach().clone() for p in module.parameters()]


def test_no_update_during_warmup():
    agent = SACAgent(SIM, TRAIN, seed=0)
    _fill_agent(agent, TRAIN["warmup_steps"] - 1)
    before = _params(agent.actor)
    assert agent.update() is None
    after = _params(agent.actor)
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_update_after_warmup_changes_actor_and_critics():
    agent = SACAgent(SIM, TRAIN, seed=0)
    _fill_agent(agent, TRAIN["warmup_steps"])
    actor_before = _params(agent.actor)
    critic_before = _params(agent.critic1)

    losses = agent.update()

    assert losses is not None
    assert np.isfinite(losses["critic_loss"])
    assert np.isfinite(losses["actor_loss"])
    assert any(not torch.equal(b, a) for b, a in zip(actor_before, _params(agent.actor)))
    assert any(not torch.equal(b, a) for b, a in zip(critic_before, _params(agent.critic1)))


def test_target_critics_are_not_trained_directly():
    agent = SACAgent(SIM, TRAIN, seed=0)
    for p in list(agent.target1.parameters()) + list(agent.target2.parameters()):
        assert p.requires_grad is False


def test_soft_update_formula():
    agent = SACAgent(SIM, TRAIN, seed=0)
    tau = TRAIN["tau"]

    target_before = _params(agent.target1)
    source = _params(agent.critic1)
    agent._soft_update(agent.critic1, agent.target1)
    target_after = _params(agent.target1)

    for t0, s, t1 in zip(target_before, source, target_after):
        expected = tau * s + (1.0 - tau) * t0
        assert torch.allclose(t1, expected, atol=1e-7)


def test_deterministic_action_is_valid():
    agent = SACAgent(SIM, TRAIN, seed=0)
    state = np.full(N, 0.5, dtype=np.float32)
    w = agent.act(state, deterministic=True)

    assert w.shape == (N,)
    assert np.all(w > 0)
    assert np.isclose(w.sum(), 1.0, atol=1e-5)


def test_sampled_action_is_valid_and_never_zero():
    agent = SACAgent(SIM, TRAIN, seed=0)
    state = np.zeros(N, dtype=np.float32)
    for _ in range(50):
        w = agent.act(state)
        assert np.all(w >= TRAIN["min_weight"] * 0.999)
        assert np.isclose(w.sum(), 1.0, atol=1e-5)


def test_same_seed_gives_same_first_action():
    a1 = SACAgent(SIM, TRAIN, seed=3).act(np.zeros(N, dtype=np.float32))
    a2 = SACAgent(SIM, TRAIN, seed=3).act(np.zeros(N, dtype=np.float32))
    assert np.array_equal(a1, a2)


def test_thirty_updates_stay_finite():
    agent = SACAgent(SIM, TRAIN, seed=0)
    _fill_agent(agent, TRAIN["warmup_steps"])
    for _ in range(30):
        losses = agent.update()
        assert np.isfinite(losses["critic_loss"])
        assert np.isfinite(losses["actor_loss"])
    for p in agent.actor.parameters():
        assert torch.all(torch.isfinite(p))


def test_agent_runs_against_real_environment():
    # Short warm-up and batch so updates happen within two episodes.
    train = copy.deepcopy(TRAIN)
    train["warmup_steps"] = 20
    train["batch_size"] = 16

    env = WFQEnv(
        arrival_rates=SIM["arrival_rates"],
        capacity=SIM["queue_capacity"],
        scv=SIM["scv"],
        bandwidth=SIM["total_bandwidth"],
        decision_interval_seconds=SIM["decision_interval_s"],
        episode_length_seconds=SIM["episode_length_s"],
    )
    agent = SACAgent(SIM, train, seed=0)

    updates_done = 0
    for episode in range(2):
        state = env.reset(seed=100 + episode)
        for _ in range(env.steps_per_episode):
            w = agent.act(state)
            next_state, reward, done, _ = env.step(w)
            assert np.isfinite(reward)
            agent.store(state, w, reward, next_state, done)
            if agent.update() is not None:
                updates_done += 1
            state = next_state

    assert updates_done > 0
