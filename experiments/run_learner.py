"""Run one learner experiment from the command line (Day 3, Block 3).

Supports:
  python experiments/run_learner.py --method sac --seed 100 --episodes 200
  python experiments/run_learner.py --method sac_ewc --seed 0 --episodes 1000

Task-anchored mode is added in Block 5.
"""

import argparse
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wfq.train import train_run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run one WFQ learner experiment")
    parser.add_argument(
        "--method",
        required=True,
        choices=["sac", "sac_ewc"],
        help="sac = plain SAC, sac_ewc = SAC with EWC",
    )
    parser.add_argument("--seed", type=int, required=True, help="Run seed")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Episode count (defaults to training.episodes in config)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "base.yaml",
        help="Path to YAML settings",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Folder for the output CSV",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    sim = cfg["simulator"]
    train = cfg["training"]
    episodes = args.episodes if args.episodes is not None else train["episodes"]

    ewc = cfg["ewc"] if args.method == "sac_ewc" else None
    out_file = args.output / f"{args.method}_seed{args.seed}.csv"

    print(
        f"Starting {args.method} run: seed={args.seed}, episodes={episodes}, "
        f"output={out_file}"
    )
    train_run(sim, train, ewc, args.seed, episodes, out_file)
    print(f"Finished {args.method} run: wrote {out_file}")


if __name__ == "__main__":
    main()
