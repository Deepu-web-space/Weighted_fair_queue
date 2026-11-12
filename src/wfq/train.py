"""Training loop for one run: SAC alone, or SAC with EWC (Day 3, Block 3).

Each episode has 100 decisions. At each decision the agent picks weights,
the environment runs 20 s, the agent stores the step and does one update
(once warm-up is over). At the end of every fisher_refresh_episodes
episodes, EWC saves new Fisher values and a new anchor.

One CSV row is written per episode. Penalties are logged every episode,
and show nan while no update has run yet.
"""

import csv
from pathlib import Path

import numpy as np

from wfq.env import WFQEnv
from wfq.sac import SACAgent


COLUMNS = [
    "episode",
    "mean_reward",
    "updates",
    "critic_loss",
    "actor_loss",
    "critic_penalty",
    "actor_penalty",
]


def make_env(sim_config):
    return WFQEnv(
        arrival_rates=sim_config["arrival_rates"],
        capacity=sim_config["queue_capacity"],
        scv=sim_config["scv"],
        bandwidth=sim_config["total_bandwidth"],
        decision_interval_seconds=sim_config["decision_interval_s"],
        episode_length_seconds=sim_config["episode_length_s"],
    )


def episode_seed(run_seed, episode_index, stride):
    """Same traffic for every method at the same run seed and episode."""
    return run_seed * stride + episode_index


def train_run(sim_config, train_config, ewc_config, seed, episodes, out_csv):
    env = make_env(sim_config)
    agent = SACAgent(sim_config, train_config, seed, ewc_config=ewc_config)
    stride = train_config["episode_seed_stride"]
    refresh_every = ewc_config["fisher_refresh_episodes"] if agent.ewc_on else None

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

        for ep in range(episodes):
            state = env.reset(seed=episode_seed(seed, ep, stride))
            rewards = []
            stats = []

            for _ in range(env.steps_per_episode):
                w = agent.act(state)
                next_state, reward, done, _ = env.step(w)
                agent.store(state, w, reward, next_state, done)
                info = agent.update()
                if info is not None:
                    stats.append(info)
                rewards.append(reward)
                state = next_state

            # Episode numbers in the file start at 1, so the refresh
            # happens after episodes refresh_every, 2 x refresh_every, ...
            if agent.ewc_on and (ep + 1) % refresh_every == 0:
                agent.refresh_ewc()

            row = {
                "episode": ep + 1,
                "mean_reward": float(np.mean(rewards)),
                "updates": len(stats),
            }
            for key in ("critic_loss", "actor_loss", "critic_penalty", "actor_penalty"):
                row[key] = float(np.mean([s[key] for s in stats])) if stats else float("nan")
            writer.writerow(row)
            f.flush()
