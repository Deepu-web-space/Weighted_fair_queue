"""analysis/make_plots.py: publication-quality figures for WFQ continual-DRL.

Generates 3 figures saved to --output-dir (default results/figures):
  1. fig1_learning_curves.png     (Stationary SAC vs SAC+EWC reward curves with shaded std)
  2. fig2_traffic_switch.png      (Retention vs forgetting bar chart across 3 methods)
  3. fig3_switch_trajectories.png (Full switch trajectory with phase switch boundary)
"""

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Clean palette: blue for SAC, amber for SAC+EWC, emerald green for Task-Anchored
COLOR_SAC = "#2563EB"           # Royal Blue
COLOR_SAC_EWC = "#D97706"       # Amber
COLOR_TASK_ANCHORED = "#059669" # Emerald


def plot_learning_curves(results_dir, output_dir):
    """Figure 1: Stationary SAC vs SAC+EWC mean reward with shaded std band."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    has_data = False
    methods = [("sac", "Plain SAC", COLOR_SAC), ("sac_ewc", "SAC with EWC (paper)", COLOR_SAC_EWC)]

    for method, label, color in methods:
        paths = sorted(results_dir.glob(f"{method}_seed*.csv"))
        paths = [p for p in paths if "_episodes" not in p.name and "_evals" not in p.name]
        if not paths:
            continue

        dfs = []
        for p in paths:
            try:
                df = pd.read_csv(p)
                dfs.append(df.set_index("episode")["mean_reward"])
            except Exception:
                continue

        if not dfs:
            continue

        has_data = True
        merged = pd.concat(dfs, axis=1)
        mean_curve = merged.mean(axis=1)
        std_curve = merged.std(axis=1).fillna(0.0)

        # Smooth curve slightly if long run
        window = 10 if len(mean_curve) > 50 else 1
        mean_smooth = mean_curve.rolling(window, min_periods=1).mean()
        std_smooth = std_curve.rolling(window, min_periods=1).mean()

        ax.plot(mean_smooth.index, mean_smooth.values, label=label, color=color, linewidth=2.0)
        ax.fill_between(mean_smooth.index,
                        mean_smooth.values - std_smooth.values,
                        mean_smooth.values + std_smooth.values,
                        color=color, alpha=0.18)

    if not has_data:
        plt.close(fig)
        return False

    ax.set_title("Stationary Learning Progression (5 Seeds)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Training Episode", fontsize=11, labelpad=8)
    ax.set_ylabel("Mean Reward per Decision", fontsize=11, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=10)
    fig.tight_layout()

    out_path = output_dir / "fig1_learning_curves.png"
    fig.savefig(out_path)
    plt.close(fig)
    print("Saved:", out_path)
    return True


def plot_traffic_switch_bars(results_dir, output_dir):
    """Figure 2: Grouped bar chart of evaluation rewards before and after switch."""
    records = []
    methods = [
        ("sac", "Plain SAC", COLOR_SAC),
        ("sac_ewc", "SAC + EWC (paper)", COLOR_SAC_EWC),
        ("task_anchored", "Task-Anchored EWC (ours)", COLOR_TASK_ANCHORED),
    ]

    for method, label, color in methods:
        for path in sorted(results_dir.glob(f"{method}_seed*_evals.csv")):
            stem = path.stem.replace("_evals", "")
            seed_str = stem.split("seed")[-1]
            try:
                seed = int(seed_str)
            except ValueError:
                continue
            df = pd.read_csv(path)
            df["method"] = method
            df["display_label"] = label
            df["color"] = color
            df["seed"] = seed
            records.append(df)

    if not records:
        return False

    all_evals = pd.concat(records, ignore_index=True)
    # Average across eval_index per seed and condition
    seed_means = (
        all_evals.groupby(["method", "display_label", "color", "seed", "checkpoint", "traffic"])["mean_reward"]
        .mean()
        .reset_index()
    )

    conditions = [
        (("end_A", "A"), "Before Switch\n(Traffic A)"),
        (("end_B", "A"), "After Switch\n(Traffic A - Retention)"),
        (("end_B", "B"), "After Switch\n(Traffic B - Adapted)"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    n_groups = len(conditions)
    bar_width = 0.24
    indices = np.arange(n_groups)

    present_methods = [m for m in methods if m[0] in seed_means["method"].unique()]

    for i, (m_key, m_label, m_color) in enumerate(present_methods):
        means = []
        stds = []
        m_data = seed_means[seed_means["method"] == m_key]
        for (ckpt, traf), _ in conditions:
            sub = m_data[(m_data["checkpoint"] == ckpt) & (m_data["traffic"] == traf)]
            vals = sub["mean_reward"].dropna()
            means.append(vals.mean() if not vals.empty else 0.0)
            stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)

        pos = indices + (i - (len(present_methods) - 1) / 2) * bar_width
        ax.bar(pos, means, bar_width, yerr=stds, capsize=4,
               label=m_label, color=m_color, edgecolor="black", linewidth=0.6, alpha=0.88)

    ax.set_title("Continual Learning and Forgetting Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Mean Evaluation Reward (closer to 0 is better)", fontsize=11, labelpad=8)
    ax.set_xticks(indices)
    ax.set_xticklabels([c[1] for c in conditions], fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=10)
    fig.tight_layout()

    out_path = output_dir / "fig2_traffic_switch.png"
    fig.savefig(out_path)
    plt.close(fig)
    print("Saved:", out_path)
    return True


def plot_switch_trajectories(results_dir, output_dir):
    """Figure 3: Episode reward trajectories during traffic switch test."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    has_data = False
    methods = [
        ("sac", "Plain SAC", COLOR_SAC),
        ("sac_ewc", "SAC + EWC (paper)", COLOR_SAC_EWC),
        ("task_anchored", "Task-Anchored EWC (ours)", COLOR_TASK_ANCHORED),
    ]

    phase_a_len = None

    for method, label, color in methods:
        paths = sorted(results_dir.glob(f"{method}_seed*_episodes.csv"))
        if not paths:
            continue

        dfs = []
        for p in paths:
            try:
                df = pd.read_csv(p)
                dfs.append(df.set_index("episode")["mean_reward"])
                if phase_a_len is None and "phase" in df.columns:
                    phase_a_len = (df["phase"] == "A").sum()
            except Exception:
                continue

        if not dfs:
            continue

        has_data = True
        merged = pd.concat(dfs, axis=1)
        mean_curve = merged.mean(axis=1)

        window = 10 if len(mean_curve) > 50 else 1
        smooth = mean_curve.rolling(window, min_periods=1).mean()
        ax.plot(smooth.index, smooth.values, label=label, color=color, linewidth=1.8)

    if not has_data:
        plt.close(fig)
        return False

    if phase_a_len and phase_a_len > 0:
        ax.axvline(x=phase_a_len, color="#DC2626", linestyle=":", linewidth=1.8, label="Traffic Switch (A to B)")

    ax.set_title("Traffic Switch Training Trajectory (5 Seeds)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Total Episode Number", fontsize=11, labelpad=8)
    ax.set_ylabel("Mean Reward per Decision", fontsize=11, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=10)
    fig.tight_layout()

    out_path = output_dir / "fig3_switch_trajectories.png"
    fig.savefig(out_path)
    plt.close(fig)
    print("Saved:", out_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures from WFQ experiment results")
    parser.add_argument("--results-dir", type=Path, default=Path("results/final"),
                        help="Folder containing experiment CSVs (default results/final)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"),
                        help="Folder to save generated figures (default results/figures)")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    print(f"Saving figures to:   {output_dir}")

    f1 = plot_learning_curves(results_dir, output_dir)
    f2 = plot_traffic_switch_bars(results_dir, output_dir)
    f3 = plot_switch_trajectories(results_dir, output_dir)

    count = sum([bool(f1), bool(f2), bool(f3)])
    print(f"\nCompleted: generated {count} of 3 figures in {output_dir}")


if __name__ == "__main__":
    main()
