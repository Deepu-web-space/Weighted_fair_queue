"""Print and plot the learning curve of one run (Day 3, Block 4).

Usage:
  python experiments/check_learning.py results\\sac_seed100.csv

It prints the average reward for each block of episodes, and saves a PNG
next to the CSV. It does not decide pass or fail. You read the numbers and
apply the stop rule from the Block 4 instructions.
"""

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [
            {
                "episode": int(r["episode"]),
                "reward": float(r["mean_reward"]),
                "updates": int(r["updates"]),
            }
            for r in csv.DictReader(f)
        ]


def block_summary(rows, window):
    """List of (first_episode, last_episode, mean_reward, sd) per block."""
    out = []
    for start in range(0, len(rows), window):
        chunk = rows[start:start + window]
        values = [r["reward"] for r in chunk]
        sd = statistics.pstdev(values) if len(values) > 1 else 0.0
        out.append((chunk[0]["episode"], chunk[-1]["episode"],
                    statistics.fmean(values), sd))
    return out


def main():
    parser = argparse.ArgumentParser(description="Check one learning curve")
    parser.add_argument("csv", type=Path, help="Path to a run CSV")
    parser.add_argument("--window", type=int, default=None,
                        help="Episodes per block (default: training.sanity_window_episodes)")
    args = parser.parse_args()

    cfg = yaml.safe_load((PROJECT_ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
    window = args.window or cfg["training"]["sanity_window_episodes"]

    rows = load_rows(args.csv)
    first_update = next((r["episode"] for r in rows if r["updates"] > 0), None)
    blocks = block_summary(rows, window)

    print(f"File: {args.csv}")
    print(f"Episodes: {len(rows)}   Block size: {window}   First episode with updates: {first_update}")
    print()
    print(f"{'episodes':>12}  {'mean reward':>12}  {'spread (sd)':>12}")
    for first, last, mean, sd in blocks:
        print(f"{first:>5}-{last:<6}  {mean:>12.3f}  {sd:>12.3f}")
    print()

    learning_blocks = [b for b in blocks if b[0] >= (first_update or 1)]
    if len(learning_blocks) >= 2:
        first_mean = learning_blocks[0][2]
        last_mean = learning_blocks[-1][2]
        best = max(learning_blocks, key=lambda b: b[2])
        print(f"First block with updates: {first_mean:.3f}")
        print(f"Last block:               {last_mean:.3f}")
        print(f"Change, first to last:    {last_mean - first_mean:+.3f}")
        print(f"Best block: episodes {best[0]}-{best[1]} at {best[2]:.3f}")
    print()
    print("Stop rule: judge these numbers against the Block 4 instructions.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    episodes = [r["episode"] for r in rows]
    rewards = [r["reward"] for r in rows]
    ax.plot(episodes, rewards, color="#B9B2A6", linewidth=0.8, label="each episode")
    centres = [(b[0] + b[1]) / 2 for b in blocks]
    ax.plot(centres, [b[2] for b in blocks], color="#8A522B", marker="o",
            linewidth=2, label=f"average per {window} episodes")
    if first_update:
        ax.axvline(first_update, color="#1F2A36", linestyle="--", linewidth=1,
                   label="learning starts")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean reward per decision")
    ax.set_title(args.csv.stem)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = args.csv.with_suffix(".png")
    fig.savefig(png, dpi=130)
    print(f"Saved plot: {png}")


if __name__ == "__main__":
    main()
