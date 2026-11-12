"""Run equal and proportional baselines across multiple random seeds.

Usage:
    python experiments/run_baselines.py
    python experiments/run_baselines.py --episodes 20 --output results/timing_test
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import yaml

# Add src/ to the Python path so the project package can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wfq.baselines import equal_weights, proportional_weights
from wfq.env import WFQEnv
from wfq.metrics import jains_index


def load_settings(config_path):
    """Load simulator and training settings from the base YAML file."""
    text = Path(config_path).read_text(encoding="utf-8")
    return yaml.safe_load(text)


def make_env(sim_config):
    """Build a WFQEnv instance using settings from the simulator section."""
    return WFQEnv(
        arrival_rates=sim_config["arrival_rates"],
        capacity=sim_config["queue_capacity"],
        scv=sim_config["scv"],
        bandwidth=sim_config["total_bandwidth"],
        decision_interval_seconds=sim_config["decision_interval_s"],
        episode_length_seconds=sim_config["episode_length_s"],
    )


def derive_episode_seed(run_seed, episode_index, max_episodes=100000):
    """Derive a reproducible seed for one episode.

    Different run seeds produce non-overlapping sequences.
    """
    return int(run_seed) * max_episodes + int(episode_index)


def run_one_episode(env, weights, episode_seed):
    """Run one full episode with fixed weights.

    Returns a dictionary of summary metrics for the episode.
    """
    env.reset(episode_seed)

    rewards = []
    total_steps = env.steps_per_episode

    for _ in range(total_steps):
        _state, reward, _done, _info = env.step(weights)
        rewards.append(reward)

    # Collect queue-level totals from the simulation
    arrived_per_queue = [q.arrived for q in env.system.queues]
    served_per_queue = [q.served for q in env.system.queues]
    lost_per_queue = [q.lost for q in env.system.queues]

    total_arrived = sum(arrived_per_queue)
    total_served = sum(served_per_queue)
    total_lost = sum(lost_per_queue)

    loss_rate = total_lost / total_arrived if total_arrived > 0 else 0.0

    # Calculate fairness across queues using the fraction of arrivals accepted
    accepted_fractions = []
    for arrived, lost in zip(arrived_per_queue, lost_per_queue):
        if arrived > 0:
            accepted_fractions.append((arrived - lost) / arrived)
        else:
            accepted_fractions.append(1.0)

    fairness = jains_index(accepted_fractions)

    result = {
        "mean_reward": sum(rewards) / len(rewards),
        "sum_reward": sum(rewards),
        "total_arrived": total_arrived,
        "total_served": total_served,
        "total_lost": total_lost,
        "loss_rate": loss_rate,
        "jains_index": fairness,
    }

    # Add per-queue loss counts
    for i, lost in enumerate(lost_per_queue, start=1):
        result[f"lost_q{i}"] = lost

    return result


def run_baseline_method(method_name, weights, env, seeds, episodes, output_csv):
    """Run a baseline across all seeds and episodes, writing rows to CSV."""
    n_queues = env.n_queues

    fieldnames = [
        "method",
        "seed",
        "episode",
        "mean_reward",
        "sum_reward",
        "total_arrived",
        "total_served",
        "total_lost",
        "loss_rate",
        "jains_index",
    ] + [f"lost_q{i}" for i in range(1, n_queues + 1)]

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Running {method_name} baseline ---")
    print(f"Seeds: {seeds}")
    print(f"Episodes per seed: {episodes}")
    print(f"Saving to: {output_csv}")

    start_time = time.time()
    total_rows = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for seed in seeds:
            seed_start = time.time()
            for ep in range(episodes):
                ep_seed = derive_episode_seed(seed, ep)
                metrics = run_one_episode(env, weights, ep_seed)

                row = {
                    "method": method_name,
                    "seed": seed,
                    "episode": ep,
                    **metrics,
                }
                writer.writerow(row)
                total_rows += 1

                # Progress indicator every 100 episodes
                if (ep + 1) % 100 == 0 or (ep + 1) == episodes:
                    elapsed = time.time() - seed_start
                    eps_per_sec = (ep + 1) / elapsed if elapsed > 0 else 0
                    print(
                        f"  seed {seed} | episode {ep + 1:4d}/{episodes} | "
                        f"mean reward: {metrics['mean_reward']:6.2f} | "
                        f"lost: {metrics['total_lost']:4d} | "
                        f"fairness: {metrics['jains_index']:.4f} | "
                        f"{eps_per_sec:5.1f} ep/s"
                    )

    elapsed_total = time.time() - start_time
    print(
        f"Completed {method_name}: {total_rows} episodes in {elapsed_total:.1f}s "
        f"({total_rows / elapsed_total:.1f} ep/s)"
    )
    return elapsed_total


def main():
    parser = argparse.ArgumentParser(
        description="Run equal and proportional WFQ baselines."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "base.yaml"),
        help="Path to base configuration file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Episodes per seed (default: from config)",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results"),
        help="Directory to save CSV outputs",
    )
    parser.add_argument(
        "--method",
        choices=["equal", "proportional", "all"],
        default="all",
        help="Which baseline to run (default: all)",
    )

    args = parser.parse_args()

    config = load_settings(args.config)
    sim_cfg = config["simulator"]
    train_cfg = config["training"]

    episodes = args.episodes if args.episodes is not None else train_cfg["episodes"]
    seeds = train_cfg["seeds_final"]
    n_queues = sim_cfg["n_queues"]
    arrival_rates = sim_cfg["arrival_rates"]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save a snapshot of the config used
    config_snapshot = output_dir / "base_config_used.yaml"
    config_snapshot.write_text(
        Path(args.config).read_text(encoding="utf-8"), encoding="utf-8"
    )

    env = make_env(sim_cfg)

    methods_to_run = []
    if args.method in ("equal", "all"):
        eq_w = equal_weights(n_queues)
        methods_to_run.append(
            ("equal", eq_w, output_dir / "baselines_equal.csv")
        )
    if args.method in ("proportional", "all"):
        prop_w = proportional_weights(arrival_rates)
        methods_to_run.append(
            ("proportional", prop_w, output_dir / "baselines_proportional.csv")
        )

    print("========================================")
    print(" WFQ Project - Step 6 Baseline Runner   ")
    print("========================================")
    print(f"Queues: {n_queues}")
    print(f"Arrival rates: {arrival_rates} (sum = {sum(arrival_rates):.4f})")
    print(f"Seeds: {seeds}")
    print(f"Episodes per seed: {episodes}")
    print(f"Total episodes to run: {len(methods_to_run) * len(seeds) * episodes}")
    print(f"Output directory: {output_dir}")

    total_start = time.time()
    for name, weights, csv_path in methods_to_run:
        run_baseline_method(name, weights, env, seeds, episodes, csv_path)

    total_time = time.time() - total_start
    print("\n========================================")
    print(f" All baselines finished in {total_time:.1f}s")
    print(f" Results saved to {output_dir}")
    print("========================================")


if __name__ == "__main__":
    main()
