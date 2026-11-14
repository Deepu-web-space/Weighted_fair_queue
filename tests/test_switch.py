"""Tests for the traffic-switch test (Day 3, Block 5).

Checks:
- Phase B traffic has the same total load and queue count as phase A.
- Each method writes both CSV files with the right rows and phase labels.
- The refresh rule: plain SAC never refreshes; the paper method refreshes
  on its schedule; task-anchored takes exactly one snapshot, at the end of phase A.
- Evaluation seeds never overlap training seeds.
"""

import copy
import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from wfq.switch import EVAL_COLUMNS, EPISODE_COLUMNS, eval_seed, run_switch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
SIM = CONFIG["simulator"]
TRAIN = CONFIG["training"]
EWC = CONFIG["ewc"]
SWITCH = CONFIG["switch_test"]


def _small_settings():
    train = copy.deepcopy(TRAIN)
    train["warmup_steps"] = 20
    train["batch_size"] = 16
    switch = copy.deepcopy(SWITCH)
    switch["eval_episodes"] = 1
    return train, switch


def test_phase_b_has_same_load_and_queue_count():
    a = SIM["arrival_rates"]
    b = SWITCH["phase_b_arrival_rates"]
    assert len(b) == len(a) == SIM["n_queues"]
    assert all(rate > 0 for rate in b)
    assert np.isclose(sum(b), sum(a))
    assert b != a  # the traffic really changes


@pytest.mark.parametrize("method", ["sac", "sac_ewc", "task_anchored"])
def test_method_writes_both_files_with_phase_labels(tmp_path, method):
    train, switch = _small_settings()
    result = run_switch(SIM, train, EWC, switch, method, seed=0,
                        phase_a_episodes=2, phase_b_episodes=2, out_dir=tmp_path)

    with result["episodes_csv"].open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == EPISODE_COLUMNS
        rows = list(reader)
    assert [int(r["episode"]) for r in rows] == [1, 2, 3, 4]
    assert [r["phase"] for r in rows] == ["A", "A", "B", "B"]

    with result["evals_csv"].open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == EVAL_COLUMNS
        evals = list(reader)
    # 3 checkpoints (end_A on A; end_B on A; end_B on B), each with 1 episode
    assert [(e["checkpoint"], e["traffic"]) for e in evals] == [
        ("end_A", "A"), ("end_B", "A"), ("end_B", "B")]
    assert all(np.isfinite(float(e["mean_reward"])) for e in evals)


def test_refresh_rule_per_method(tmp_path):
    train, switch = _small_settings()
    ewc_every_episode = copy.deepcopy(EWC)
    ewc_every_episode["fisher_refresh_episodes"] = 1

    plain = run_switch(SIM, train, None, switch, "sac", 0, 2, 2, tmp_path / "sac")
    paper = run_switch(SIM, train, ewc_every_episode, switch, "sac_ewc", 0, 2, 2,
                       tmp_path / "paper")
    anchored = run_switch(SIM, train, ewc_every_episode, switch, "task_anchored", 0, 2, 2,
                          tmp_path / "anchored")

    assert plain["refresh_episodes"] == []
    assert paper["refresh_episodes"] == [1, 2, 3, 4]   # every episode, including phase B
    assert anchored["refresh_episodes"] == [2]          # once, at the end of phase A


def test_task_anchored_has_no_penalty_in_phase_a(tmp_path):
    train, switch = _small_settings()
    result = run_switch(SIM, train, EWC, switch, "task_anchored", 0, 2, 2, tmp_path)
    with result["episodes_csv"].open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # No anchor exists during phase A, so the penalty is exactly zero there.
    assert float(rows[0]["actor_penalty"]) == 0.0
    assert float(rows[1]["actor_penalty"]) == 0.0


def test_eval_seeds_do_not_overlap_training_seeds():
    base = SWITCH["eval_seed_base"]
    stride = SWITCH["eval_seed_stride"]
    last_training_seed = max(
        s * TRAIN["episode_seed_stride"] + (SWITCH["phase_a_episodes"]
                                            + SWITCH["phase_b_episodes"] - 1)
        for s in TRAIN["seeds_final"] + TRAIN["seeds_tuning"]
    )
    first_eval_seed = min(
        eval_seed(s, k, base, stride)
        for s in TRAIN["seeds_final"] + TRAIN["seeds_tuning"]
        for k in range(SWITCH["eval_episodes"])
    )
    assert first_eval_seed > last_training_seed
    assert SWITCH["eval_episodes"] <= stride
