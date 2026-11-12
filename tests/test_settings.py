"""Tests for Day 3, Block 1: the settings the SAC and EWC learner will read.

The expected values below are the paper's Table 1 values (Mawlood and Mahmood,
Telecom 2025, 6, 46). If a value here changes, the change must be logged in
docs/decisions.md first.
"""

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]

CONFIG = yaml.safe_load(
    (ROOT / "config" / "base.yaml").read_text(encoding="utf-8")
)

SIM = CONFIG["simulator"]
TRAIN = CONFIG["training"]
EWC = CONFIG["ewc"]
SWITCH = CONFIG["switch_test"]


def test_all_sections_present():
    # The learner reads four sections. A missing one must fail loudly.
    for section in ("simulator", "training", "ewc", "switch_test"):
        assert section in CONFIG


def test_sac_values_match_paper_table_1():
    # Paper Table 1: discount, soft update, entropy weight, learning rates.
    assert TRAIN["gamma"] == pytest.approx(0.99)
    assert TRAIN["tau"] == pytest.approx(0.01)
    assert TRAIN["alpha"] == pytest.approx(0.2)
    assert TRAIN["lr_actor"] == pytest.approx(1e-4)
    assert TRAIN["lr_critic"] == pytest.approx(1e-3)


def test_replay_and_batch_match_paper_table_1():
    # Paper Table 1, 8-queue setting.
    assert TRAIN["replay_size"] == 10000
    assert TRAIN["batch_size"] == 128


def test_ewc_values_match_paper_8_queues():
    # Paper Table 1: lambda 2271 for 8 queues; Fisher refresh every 30 episodes.
    assert EWC["lambda"] == 2271
    assert EWC["fisher_refresh_episodes"] == 30


def test_ewc_mode_is_a_known_value():
    # Only the two modes in the decision log are allowed.
    assert EWC["mode"] in ("paper", "task_anchored")


def test_queue_count_matches_arrival_rates():
    # One arrival rate per queue, or the environment will not start.
    assert len(SIM["arrival_rates"]) == SIM["n_queues"]


def test_episode_has_100_decisions():
    # 2000 s episode, decision every 20 s: 100 decisions per episode.
    steps = SIM["episode_length_s"] / SIM["decision_interval_s"]
    assert steps == pytest.approx(100)


def test_switch_phases_are_500_episodes_each():
    assert SWITCH["phase_a_episodes"] == 500
    assert SWITCH["phase_b_episodes"] == 500


def test_tuning_and_final_seeds_do_not_overlap():
    # Rule: never tune on the seeds used for final results.
    final = set(TRAIN["seeds_final"])
    tuning = set(TRAIN["seeds_tuning"])

    assert final == {0, 1, 2, 3, 4}
    assert tuning == {100, 101, 102, 103, 104}
    assert final.isdisjoint(tuning)



def test_torch_threads_is_set_for_parallel_runs():
    # Three or four runs at once, each with a small thread count.
    assert 1 <= TRAIN["torch_threads"] <= 4
