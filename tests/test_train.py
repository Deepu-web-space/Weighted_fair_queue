"""Tests for the training loop and its CSV logging (Day 3, Block 3)."""

import copy
import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from wfq.train import COLUMNS, train_run


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
SIM = CONFIG["simulator"]
TRAIN = CONFIG["training"]
EWC = CONFIG["ewc"]


def test_train_run_writes_valid_csv(tmp_path):
    train = copy.deepcopy(TRAIN)
    train["warmup_steps"] = 20
    train["batch_size"] = 16

    out_csv = tmp_path / "test_run.csv"
    train_run(SIM, train, ewc_config=None, seed=0, episodes=2, out_csv=out_csv)

    assert out_csv.exists()
    with out_csv.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 2
    assert list(reader[0].keys()) == COLUMNS
    assert reader[0]["episode"] == "1"
    assert reader[1]["episode"] == "2"

    for row in reader:
        assert np.isfinite(float(row["mean_reward"]))
        assert int(row["updates"]) > 0
        assert np.isfinite(float(row["critic_loss"]))
        assert np.isfinite(float(row["actor_loss"]))
        # Without EWC, penalty columns are recorded as 0.0
        assert float(row["critic_penalty"]) == 0.0
        assert float(row["actor_penalty"]) == 0.0


def test_train_run_with_ewc_logs_penalties(tmp_path):
    train = copy.deepcopy(TRAIN)
    train["warmup_steps"] = 20
    train["batch_size"] = 16
    ewc = copy.deepcopy(EWC)
    ewc["fisher_refresh_episodes"] = 1

    out_csv = tmp_path / "test_ewc_run.csv"
    train_run(SIM, train, ewc_config=ewc, seed=0, episodes=2, out_csv=out_csv)

    with out_csv.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 2
    # Before the first refresh (episode 1), the penalty is 0.
    assert float(reader[0]["actor_penalty"]) == 0.0
    # After the refresh at the end of episode 1, episode 2 has a real anchor.
    assert np.isfinite(float(reader[1]["actor_penalty"]))
    assert float(reader[1]["actor_penalty"]) >= 0.0
