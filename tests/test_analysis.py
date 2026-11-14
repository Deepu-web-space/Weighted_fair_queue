"""tests/test_analysis.py: verify table and plot generation functions."""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import pytest


@pytest.fixture
def sample_results_dir(tmp_path):
    """Create minimal valid CSVs for testing analysis scripts."""
    results = tmp_path / "results"
    results.mkdir()

    # 1. Stationary learner CSVs: sac and sac_ewc for seed 0 and 1
    for method in ["sac", "sac_ewc"]:
        for seed in [0, 1]:
            df = pd.DataFrame({
                "episode": [1, 2, 3],
                "mean_reward": [-18.5 + seed, -17.0 + seed, -16.2 + seed],
                "updates": [0, 100, 100],
                "critic_loss": [float("nan"), 150.0, 140.0],
                "actor_loss": [float("nan"), 200.0, 190.0],
                "critic_penalty": [float("nan"), 0.0, 0.5],
                "actor_penalty": [float("nan"), 0.0, 0.01],
            })
            df.to_csv(results / f"{method}_seed{seed}.csv", index=False)

    # 2. Switch test CSVs: episodes and evals
    for method in ["sac", "sac_ewc", "task_anchored"]:
        for seed in [0, 1]:
            # Episodes
            ep_df = pd.DataFrame({
                "episode": [1, 2, 3, 4],
                "phase": ["A", "A", "B", "B"],
                "mean_reward": [-18.0, -17.5, -22.0, -21.0],
                "updates": [0, 100, 100, 100],
                "critic_loss": [float("nan"), 120.0, 130.0, 125.0],
                "actor_loss": [float("nan"), 180.0, 190.0, 185.0],
                "critic_penalty": [float("nan"), 0.0, 10.0, 12.0],
                "actor_penalty": [float("nan"), 0.0, 0.05, 0.06],
            })
            ep_df.to_csv(results / f"{method}_seed{seed}_episodes.csv", index=False)

            # Evals
            ev_df = pd.DataFrame({
                "checkpoint": ["end_A", "end_A", "end_B", "end_B", "end_B"],
                "traffic": ["A", "A", "A", "A", "B"],
                "eval_index": [0, 1, 0, 1, 0],
                "mean_reward": [-12.0, -11.8, -14.5 if method == "sac" else -12.5, -14.2 if method == "sac" else -12.3, -13.0],
            })
            ev_df.to_csv(results / f"{method}_seed{seed}_evals.csv", index=False)

    return results


def test_make_tables_creates_expected_files(sample_results_dir, tmp_path):
    out_dir = tmp_path / "tables"
    script = Path(__file__).resolve().parents[1] / "analysis" / "make_tables.py"

    res = subprocess.run([
        sys.executable, str(script),
        "--results-dir", str(sample_results_dir),
        "--output-dir", str(out_dir),
    ], capture_output=True, text=True)

    assert res.returncode == 0, res.stderr
    assert (out_dir / "table1_stationary.csv").exists()
    assert (out_dir / "table1_stationary.md").exists()
    assert (out_dir / "table2_switch_forgetting.csv").exists()
    assert (out_dir / "table2_switch_forgetting.md").exists()

    # Content checks
    t2_text = (out_dir / "table2_switch_forgetting.md").read_text(encoding="utf-8")
    assert "task_anchored" in t2_text
    assert "sac" in t2_text


def test_make_plots_creates_expected_figures(sample_results_dir, tmp_path):
    out_dir = tmp_path / "figures"
    script = Path(__file__).resolve().parents[1] / "analysis" / "make_plots.py"

    res = subprocess.run([
        sys.executable, str(script),
        "--results-dir", str(sample_results_dir),
        "--output-dir", str(out_dir),
    ], capture_output=True, text=True)

    assert res.returncode == 0, res.stderr
    assert (out_dir / "fig1_learning_curves.png").exists()
    assert (out_dir / "fig2_traffic_switch.png").exists()
    assert (out_dir / "fig3_switch_trajectories.png").exists()
    assert (out_dir / "fig1_learning_curves.png").stat().st_size > 1000


def test_empty_results_directory_handled_gracefully(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    out_dir = tmp_path / "out"

    tables_script = Path(__file__).resolve().parents[1] / "analysis" / "make_tables.py"
    res1 = subprocess.run([
        sys.executable, str(tables_script),
        "--results-dir", str(empty_dir),
        "--output-dir", str(out_dir / "tables"),
    ], capture_output=True, text=True)
    assert res1.returncode == 0

    plots_script = Path(__file__).resolve().parents[1] / "analysis" / "make_plots.py"
    res2 = subprocess.run([
        sys.executable, str(plots_script),
        "--results-dir", str(empty_dir),
        "--output-dir", str(out_dir / "figures"),
    ], capture_output=True, text=True)
    assert res2.returncode == 0
