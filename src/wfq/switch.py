"""Traffic-switch test (Day 3, Block 5).

The learner trains on traffic A for phase_a_episodes, then on traffic B for
phase_b_episodes. Forgetting is checked by replaying traffic A after each
phase with the learned policy (no learning during these checks).

Three methods:
- sac: plain SAC, no memory penalty (control).
- sac_ewc: the paper's method. The anchor is refreshed every
  fisher_refresh_episodes episodes, including during phase B.
- task_anchored: our change. No penalty during phase A. One fixed snapshot
  is taken at the end of phase A and kept unchanged through phase B.

Checkpoints written to the evaluation CSV:
- end_A on traffic A (how well the learner does on A after phase A)
- end_B on traffic A (the same traffic after phase B; the drop is forgetting)
- end_B on traffic B (how well the learner does on the new traffic)
"""


import csv
from pathlib import Path

import numpy as np

from wfq.sac import SACAgent
from wfq.train import episode_seed, make_env


METHODS = ("sac", "sac_ewc", "task_anchored")

EPISODE_COLUMNS = [
    "episode",
    "phase",
    "mean_reward",
    "updates",
    "critic_loss",
    "actor_loss",
    "critic_penalty",
    "actor_penalty",
]
EVAL_COLUMNS = ["checkpoint", "traffic", "eval_index", "mean_reward"]


def eval_seed(run_seed, eval_index, eval_seed_base, eval_seed_stride):
    """Fixed evaluation seed. Same seeds for every method, so checks are paired."""
    return eval_seed_base + run_seed * eval_seed_stride + eval_index


def _train_episode(env, agent, seed):
    state = env.reset(seed=seed)
    rewards, stats = [], []
    for _ in range(env.steps_per_episode):
        w = agent.act(state)
        next_state, reward, done, _ = env.step(w)
        agent.store(state, w, reward, next_state, done)
        info = agent.update()
        if info is not None:
            stats.append(info)
        rewards.append(reward)
        state = next_state
    return rewards, stats


def _evaluate(agent, sim_config, rates, run_seed, switch_cfg):
    """Mean reward of the learned policy on fixed episodes. No learning happens here."""
    env = make_env({**sim_config, "arrival_rates": rates})
    results = []
    for k in range(switch_cfg["eval_episodes"]):
        state = env.reset(seed=eval_seed(
            run_seed, k, switch_cfg["eval_seed_base"], switch_cfg["eval_seed_stride"]))
        rewards = []
        for _ in range(env.steps_per_episode):
            w = agent.act(state, deterministic=True)
            state, reward, _, _ = env.step(w)
            rewards.append(reward)
        results.append(float(np.mean(rewards)))
    return results


def run_switch(sim_config, train_config, ewc_config, switch_cfg, method, seed,
               phase_a_episodes, phase_b_episodes, out_dir):
    """Run one switch test. Returns the list of episode numbers where a refresh happened."""
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")

    total = phase_a_episodes + phase_b_episodes
    if total >= train_config["episode_seed_stride"]:
        raise ValueError("too many episodes for the training seed stride")
    if switch_cfg["eval_episodes"] > switch_cfg["eval_seed_stride"]:
        raise ValueError("eval_episodes must not exceed eval_seed_stride")

    # The method name decides the behaviour. The ewc.mode key is not read by the code.
    agent_ewc = None if method == "sac" else ewc_config

    env_a = make_env({**sim_config, "arrival_rates": sim_config["arrival_rates"]})
    env_b = make_env({**sim_config, "arrival_rates": switch_cfg["phase_b_arrival_rates"]})
    agent = SACAgent(sim_config, train_config, seed, ewc_config=agent_ewc)
    stride = train_config["episode_seed_stride"]
    refresh_every = ewc_config["fisher_refresh_episodes"] if agent.ewc_on else None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_csv = out_dir / f"{method}_seed{seed}_episodes.csv"
    evals_csv = out_dir / f"{method}_seed{seed}_evals.csv"

    refresh_episodes = []
    eval_rows = []

    def checkpoint(name):
        for traffic, rates in (("A", sim_config["arrival_rates"]),
                               ("B", switch_cfg["phase_b_arrival_rates"])):
            if name == "end_A" and traffic == "B":
                continue
            for k, value in enumerate(_evaluate(agent, sim_config, rates, seed, switch_cfg)):
                eval_rows.append({"checkpoint": name, "traffic": traffic,
                                  "eval_index": k, "mean_reward": value})

    with episodes_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_COLUMNS)
        writer.writeheader()

        for ep in range(total):
            number = ep + 1
            in_a = ep < phase_a_episodes
            env = env_a if in_a else env_b
            rewards, stats = _train_episode(env, agent, episode_seed(seed, ep, stride))

            if agent.ewc_on:
                if method == "sac_ewc" and number % refresh_every == 0:
                    agent.refresh_ewc()
                    refresh_episodes.append(number)
                if method == "task_anchored" and number == phase_a_episodes:
                    agent.refresh_ewc()
                    refresh_episodes.append(number)

            row = {
                "episode": number,
                "phase": "A" if in_a else "B",
                "mean_reward": float(np.mean(rewards)),
                "updates": len(stats),
            }
            for key in ("critic_loss", "actor_loss", "critic_penalty", "actor_penalty"):
                row[key] = float(np.mean([s[key] for s in stats])) if stats else float("nan")
            writer.writerow(row)
            f.flush()

            if number == phase_a_episodes:
                checkpoint("end_A")
            if number == total:
                checkpoint("end_B")

    with evals_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_COLUMNS)
        writer.writeheader()
        writer.writerows(eval_rows)

    return {"refresh_episodes": refresh_episodes,
            "episodes_csv": episodes_csv,
            "evals_csv": evals_csv}
