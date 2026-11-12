"""Gate tests for Day 3, Block 3 (the EWC module).

Three tests match the Day 3 plan's gate checks:
(a) With lambda = 0, SAC with EWC matches plain SAC exactly for the same seed.
(b) The penalty is zero right after a refresh.
(c) No Fisher value is ever negative.

Plus one hand-calculated penalty check on a tiny module, and one check that
the penalty is zero before the first refresh.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from wfq.ewc import EWCAnchor
from wfq.sac import SACAgent


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
SIM = CONFIG["simulator"]
TRAIN = CONFIG["training"]
EWC = CONFIG["ewc"]
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


def test_hand_calculated_penalty():
    # p starts at [1.0, 2.0], loss = (p[0]**2 + p[1]**2),
    # so gradient is [2.0, 4.0] and squared gradient is [4.0, 16.0].
    # After moving p to [2.0, 2.0], diff is [1.0, 0.0].
    # penalty = 0.5 * lambda * (4.0 * 1^2 + 16.0 * 0^2) = 0.5 * 3.0 * 4.0 = 6.0.
    anchor = EWCAnchor(lambda_ewc=3.0)
    p = nn.Parameter(torch.tensor([1.0, 2.0]))
    anchor.snapshot([p], lambda: (p[0] ** 2 + p[1] ** 2), n_batches=1)

    with torch.no_grad():
        p.add_(torch.tensor([1.0, 0.0]))

    pen = anchor.penalty([p])
    assert torch.isclose(pen, torch.tensor(6.0))


def test_penalty_is_zero_before_first_refresh():
    agent = SACAgent(SIM, TRAIN, seed=0, ewc_config=EWC)
    _fill_agent(agent, TRAIN["warmup_steps"] + 1)
    info = agent.update()
    assert info is not None
    assert info["actor_penalty"] == 0.0
    assert info["critic_penalty"] == 0.0


def test_penalty_is_zero_right_after_refresh():
    # Gate test (b): right after a refresh the anchor matches the current weights.
    agent = SACAgent(SIM, TRAIN, seed=0, ewc_config=EWC)
    _fill_agent(agent, TRAIN["warmup_steps"] + TRAIN["batch_size"])
    agent.refresh_ewc()

    pen_actor = agent.ewc_actor.penalty(list(agent.actor.parameters()))
    pen_critic = agent.ewc_critic.penalty(agent._critic_params())
    assert torch.isclose(pen_actor, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(pen_critic, torch.tensor(0.0), atol=1e-6)


def test_fisher_values_are_never_negative():
    # Gate test (c): Fisher is an average of squared gradients, so it must be >= 0.
    agent = SACAgent(SIM, TRAIN, seed=0, ewc_config=EWC)
    _fill_agent(agent, TRAIN["warmup_steps"] + TRAIN["batch_size"])
    agent.refresh_ewc()

    for f in agent.ewc_actor.fisher + agent.ewc_critic.fisher:
        assert torch.all(f >= 0.0)


def test_lambda_zero_matches_plain_sac_exactly():
    # Gate test (a): when lambda = 0, no extra random numbers are drawn and
    # the penalty is zero, so every parameter matches plain SAC.
    ewc_zero = dict(EWC)
    ewc_zero["lambda"] = 0

    plain = SACAgent(SIM, TRAIN, seed=42, ewc_config=None)
    _fill_agent(plain, TRAIN["warmup_steps"] + 5)
    for _ in range(5):
        plain.update()
    plain_params = [p.detach().clone() for p in plain.actor.parameters()]

    with_zero = SACAgent(SIM, TRAIN, seed=42, ewc_config=ewc_zero)
    _fill_agent(with_zero, TRAIN["warmup_steps"] + 5)
    for _ in range(5):
        with_zero.update()
    zero_params = [p.detach().clone() for p in with_zero.actor.parameters()]

    for p_plain, p_zero in zip(plain_params, zero_params):
        assert torch.equal(p_plain, p_zero)
