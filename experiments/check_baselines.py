"""Verify and summarize baseline results CSV files.

Usage:
    python experiments/check_baselines.py
    python experiments/check_baselines.py --results results
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    """Read a CSV file into a list of row dictionaries."""
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def summarize_method(rows):
    """Compute summary statistics across all episodes and seeds."""
    if not rows:
        return {}

    n = len(rows)
    seeds = sorted({int(r["seed"]) for r in rows})

    mean_reward = sum(float(r["mean_reward"]) for r in rows) / n
    mean_loss_rate = sum(float(r["loss_rate"]) for r in rows) / n
    mean_fairness = sum(float(r["jains_index"]) for r in rows) / n
    total_lost = sum(int(r["total_lost"]) for r in rows)
    total_arrived = sum(int(r["total_arrived"]) for r in rows)
    total_served = sum(int(r["total_served"]) for r in rows)

    # Per-queue losses
    queue_losses = {}
    q = 1
    while f"lost_q{q}" in rows[0]:
        col = f"lost_q{q}"
        queue_losses[col] = sum(int(r[col]) for r in rows)
        q += 1

    return {
        "n_rows": n,
        "seeds": seeds,
        "mean_reward": mean_reward,
        "mean_loss_rate": mean_loss_rate,
        "mean_fairness": mean_fairness,
        "total_lost": total_lost,
        "total_arrived": total_arrived,
        "total_served": total_served,
        "overall_loss_rate": total_lost / total_arrived if total_arrived > 0 else 0.0,
        "queue_losses": queue_losses,
    }


def main():
    parser = argparse.ArgumentParser(description="Check baseline experiment results.")
    parser.add_argument(
        "--results",
        default=str(PROJECT_ROOT / "results"),
        help="Directory containing baseline CSV files",
    )
    args = parser.parse_args()

    results_dir = Path(args.results)

    equal_csv = results_dir / "baselines_equal.csv"
    prop_csv = results_dir / "baselines_proportional.csv"

    print("========================================")
    print(" WFQ Project - Step 6 Results Check     ")
    print("========================================")
    print(f"Checking directory: {results_dir}\n")

    passed_all = True

    try:
        equal_rows = read_csv(equal_csv)
        print(f"OK: Found {equal_csv.name} ({len(equal_rows)} rows)")
    except Exception as e:
        print(f"FAIL: {e}")
        return 1

    try:
        prop_rows = read_csv(prop_csv)
        print(f"OK: Found {prop_csv.name} ({len(prop_rows)} rows)")
    except Exception as e:
        print(f"FAIL: {e}")
        return 1

    eq_summary = summarize_method(equal_rows)
    prop_summary = summarize_method(prop_rows)

    print("\n--- Summary Table ---")
    print(f"{'Metric':<28} | {'Equal':>14} | {'Proportional':>14}")
    print("-" * 62)
    print(f"{'Total episodes':<28} | {eq_summary['n_rows']:>14d} | {prop_summary['n_rows']:>14d}")
    print(f"{'Seeds tested':<28} | {str(eq_summary['seeds']):>14} | {str(prop_summary['seeds']):>14}")
    print(f"{'Mean reward (per step)':<28} | {eq_summary['mean_reward']:>14.3f} | {prop_summary['mean_reward']:>14.3f}")
    print(f"{'Total packets arrived':<28} | {eq_summary['total_arrived']:>14,d} | {prop_summary['total_arrived']:>14,d}")
    print(f"{'Total packets served':<28} | {eq_summary['total_served']:>14,d} | {prop_summary['total_served']:>14,d}")
    print(f"{'Total packets lost':<28} | {eq_summary['total_lost']:>14,d} | {prop_summary['total_lost']:>14,d}")
    print(f"{'Overall loss rate':<28} | {eq_summary['overall_loss_rate'] * 100:>13.2f}% | {prop_summary['overall_loss_rate'] * 100:>13.2f}%")
    print(f"{'Jain Fairness Index':<28} | {eq_summary['mean_fairness']:>14.4f} | {prop_summary['mean_fairness']:>14.4f}")

    print("\n--- Per-Queue Packet Loss Breakdown ---")
    print(f"{'Queue':<10} | {'Arrival rate':>14} | {'Equal lost':>14} | {'Proportional lost':>18}")
    print("-" * 64)
    q_names = sorted(eq_summary["queue_losses"].keys())
    for q_col in q_names:
        eq_lost = eq_summary["queue_losses"][q_col]
        prop_lost = prop_summary["queue_losses"].get(q_col, 0)
        print(f"{q_col:<10} | {'':>14} | {eq_lost:>14,d} | {prop_lost:>18,d}")

    # Sanity checks
    print("\n--- Sanity Checks ---")

    # Check 1: Equal baselines should lose packets on overloaded queues (q2, q3, q6)
    eq_losses = eq_summary["queue_losses"]
    overloaded_lost = (
        eq_losses.get("lost_q2", 0)
        + eq_losses.get("lost_q3", 0)
        + eq_losses.get("lost_q6", 0)
    )
    light_lost = (
        eq_losses.get("lost_q1", 0)
        + eq_losses.get("lost_q4", 0)
        + eq_losses.get("lost_q5", 0)
        + eq_losses.get("lost_q7", 0)
        + eq_losses.get("lost_q8", 0)
    )

    if overloaded_lost > 0:
        print(f"PASS: Overloaded queues (2, 3, 6) lost {overloaded_lost:,} packets under equal weights")
    else:
        print("FAIL: Expected packet loss on overloaded queues under equal weights")
        passed_all = False

    if light_lost == 0 or light_lost < 0.01 * overloaded_lost:
        print(f"PASS: Light queues (1, 4, 5, 7, 8) lost minimal packets ({light_lost:,})")
    else:
        print(f"WARN: Light queues had noticeable loss ({light_lost:,})")

    # Check 2: Proportional should have higher (less negative) reward than equal
    if prop_summary["mean_reward"] > eq_summary["mean_reward"]:
        diff = prop_summary["mean_reward"] - eq_summary["mean_reward"]
        print(f"PASS: Proportional reward is higher than equal (difference: +{diff:.2f})")
    else:
        print("FAIL: Proportional reward is not higher than equal")
        passed_all = False

    # Check 3: Proportional should lose fewer packets than equal
    if prop_summary["total_lost"] < eq_summary["total_lost"]:
        pct = (
            1.0 - prop_summary["total_lost"] / eq_summary["total_lost"]
        ) * 100.0
        print(f"PASS: Proportional reduces packet loss by {pct:.1f}% vs equal")
    else:
        print("FAIL: Proportional did not reduce packet loss vs equal")
        passed_all = False

    # Check 4: Proportional fairness should be higher
    if prop_summary["mean_fairness"] >= eq_summary["mean_fairness"]:
        print(f"PASS: Proportional fairness ({prop_summary['mean_fairness']:.4f}) >= equal ({eq_summary['mean_fairness']:.4f})")
    else:
        print("FAIL: Proportional fairness is lower than equal")
        passed_all = False

    print("\n========================================")
    if passed_all:
        print(" ALL CHECKS PASSED: Baseline results are valid!")
    else:
        print(" SOME CHECKS FAILED: Review the output above.")
    print("========================================")

    return 0 if passed_all else 1


if __name__ == "__main__":
    sys.exit(main())
