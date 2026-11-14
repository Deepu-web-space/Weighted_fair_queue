"""Run one traffic-switch test from the command line (Day 3, Block 5).

Examples:
  python experiments/run_switch.py --method task_anchored --seed 100
  python experiments/run_switch.py --method sac --seed 0 --phase-a 2 --phase-b 2 --output results/smoke

Phase lengths default to switch_test.phase_a_episodes and phase_b_episodes in config.
Do not run the final seeds (0 to 4) until Block 4 has passed.
"""

import argparse
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wfq.switch import METHODS, run_switch  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run one traffic-switch test")
    parser.add_argument("--method", required=True, choices=METHODS,
                        help="sac, sac_ewc (paper), or task_anchored (ours)")
    parser.add_argument("--seed", type=int, required=True, help="Run seed")
    parser.add_argument("--phase-a", type=int, default=None, help="Episodes on traffic A")
    parser.add_argument("--phase-b", type=int, default=None, help="Episodes on traffic B")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "base.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "switch")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    switch_cfg = cfg["switch_test"]
    a = args.phase_a if args.phase_a is not None else switch_cfg["phase_a_episodes"]
    b = args.phase_b if args.phase_b is not None else switch_cfg["phase_b_episodes"]

    print(f"Starting {args.method} switch test: seed={args.seed}, phase A={a}, phase B={b}")
    result = run_switch(cfg["simulator"], cfg["training"], cfg["ewc"], switch_cfg,
                        args.method, args.seed, a, b, args.output)
    print(f"Refresh episodes: {result['refresh_episodes']}")
    print(f"Wrote {result['episodes_csv']}")
    print(f"Wrote {result['evals_csv']}")


if __name__ == "__main__":
    main()
