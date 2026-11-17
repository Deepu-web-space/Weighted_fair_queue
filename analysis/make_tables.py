"""analysis/make_tables.py: generate summary tables from WFQ continual-DRL results.

Reads results from --results-dir (default results/final).
Outputs Markdown and CSV tables to --output-dir (default results/tables):
  1. table1_stationary.csv / .md  (SAC vs SAC+EWC reward across episodes)
  2. table2_switch_forgetting.csv / .md  (Forgetting comparison across switch methods)
  3. table3_baselines.csv / .md  (Equal, Proportional, SAC, SAC+EWC comparison)
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def load_learner_results(results_dir):
    """Load stationary learner CSVs: {method}_seed{seed}.csv"""
    records = []
    for method in ["sac", "sac_ewc"]:
        for path in sorted(results_dir.glob(f"{method}_seed*.csv")):
            if "_episodes" in path.name or "_evals" in path.name:
                continue
            seed_str = path.stem.split("seed")[-1]
            try:
                seed = int(seed_str)
            except ValueError:
                continue
            df = pd.read_csv(path)
            df["method"] = method
            df["seed"] = seed
            records.append(df)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def load_switch_evals(results_dir):
    """Load switch evaluation CSVs: {method}_seed{seed}_evals.csv"""
    records = []
    for method in ["sac", "sac_ewc", "task_anchored"]:
        for path in sorted(results_dir.glob(f"{method}_seed*_evals.csv")):
            stem = path.stem.replace("_evals", "")
            seed_str = stem.split("seed")[-1]
            try:
                seed = int(seed_str)
            except ValueError:
                continue
            df = pd.read_csv(path)
            df["method"] = method
            df["seed"] = seed
            records.append(df)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def load_baselines(results_dir, project_root):
    """Load baseline CSVs from results_dir or project results/baselines."""
    candidates = [
        results_dir,
        project_root / "results" / "baselines",
        project_root / "results",
    ]
    records = []
    for d in candidates:
        for method in ["equal", "proportional"]:
            p = d / f"baselines_{method}.csv"
            if p.exists():
                df = pd.read_csv(p)
                records.append(df)
        if records:
            break
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def build_stationary_table(learner_df):
    """Summarise stationary runs across seeds by method and episode checkpoints."""
    if learner_df.empty:
        return pd.DataFrame()

    max_ep = int(learner_df["episode"].max())
    # Choose checkpoints that fit the run length
    if max_ep <= 10:
        checkpoints = sorted(learner_df["episode"].unique().tolist())
    else:
        candidates = [50, 100, 200, 300, 400, 500, 800, 1000]
        checkpoints = [c for c in candidates if c <= max_ep]
        if max_ep not in checkpoints:
            checkpoints.append(max_ep)

    rows = []
    for method, mdf in learner_df.groupby("method"):
        for ep in checkpoints:
            sub = mdf[mdf["episode"] == ep]
            if sub.empty:
                continue
            rewards = sub["mean_reward"].dropna()
            c_loss = sub["critic_loss"].dropna()
            a_loss = sub["actor_loss"].dropna()
            rows.append({
                "Method": method,
                "Episode": ep,
                "Mean Reward": rewards.mean(),
                "Reward Std": rewards.std(ddof=1) if len(rewards) > 1 else 0.0,
                "Critic Loss": c_loss.mean() if not c_loss.empty else np.nan,
                "Actor Loss": a_loss.mean() if not a_loss.empty else np.nan,
                "Seeds Count": len(rewards),
            })
    return pd.DataFrame(rows)


def build_switch_table(evals_df):
    """Summarise traffic switch evaluation checkpoints and retention."""
    if evals_df.empty:
        return pd.DataFrame()

    # Average over eval_index within each seed and checkpoint first
    seed_means = (
        evals_df.groupby(["method", "seed", "checkpoint", "traffic"])["mean_reward"]
        .mean()
        .reset_index()
    )

    rows = []
    methods = [m for m in ["sac", "sac_ewc", "task_anchored"] if m in seed_means["method"].unique()]
    for method in methods:
        mdf = seed_means[seed_means["method"] == method]
        # Pivot by checkpoint and traffic
        piv = mdf.pivot(index="seed", columns=["checkpoint", "traffic"], values="mean_reward")

        # end_A on Traffic A
        r_a_end_a = piv[("end_A", "A")] if ("end_A", "A") in piv.columns else pd.Series(dtype=float)
        # end_B on Traffic A (retention test)
        r_a_end_b = piv[("end_B", "A")] if ("end_B", "A") in piv.columns else pd.Series(dtype=float)
        # end_B on Traffic B (adaptation to new pattern)
        r_b_end_b = piv[("end_B", "B")] if ("end_B", "B") in piv.columns else pd.Series(dtype=float)

        # Forgetting: R_A(end_A) - R_A(end_B) (positive = dropped / forgot)
        drop = r_a_end_a - r_a_end_b if not r_a_end_a.empty and not r_a_end_b.empty else pd.Series(dtype=float)

        rows.append({
            "Method": method,
            "Seeds": len(piv),
            "Before Switch (A)": f"{r_a_end_a.mean():.3f} +/- {r_a_end_a.std(ddof=1):.3f}" if len(r_a_end_a) > 1 else f"{r_a_end_a.mean():.3f}",
            "After Switch (A retained)": f"{r_a_end_b.mean():.3f} +/- {r_a_end_b.std(ddof=1):.3f}" if len(r_a_end_b) > 1 else f"{r_a_end_b.mean():.3f}",
            "Forgetting Drop (A)": f"{drop.mean():.3f} +/- {drop.std(ddof=1):.3f}" if len(drop) > 1 else f"{drop.mean():.3f}",
            "After Switch (B adapted)": f"{r_b_end_b.mean():.3f} +/- {r_b_end_b.std(ddof=1):.3f}" if len(r_b_end_b) > 1 else f"{r_b_end_b.mean():.3f}",
        })

    return pd.DataFrame(rows)


def build_baselines_table(baselines_df, learner_df):
    """Compare equal and proportional baselines with final learner performance."""
    rows = []
    if not baselines_df.empty:
        for method, bdf in baselines_df.groupby("method"):
            # Aggregate over all episodes and seeds
            rewards = bdf["mean_reward"].dropna()
            #losses = bdf["loss_rate"].dropna() if "loss_rate" in bdf.columns else pd.Series(dtype=float)
            #jain = bdf["jains_index"].dropna() if "jains_index" in bdf.columns else pd.Series(dtype=float)
            total_lost = bdf["total_lost"].sum()
            total_arrived = bdf["total_arrived"].sum()

            loss_rate_pct = (
                (total_lost / total_arrived) * 100
                if total_arrived > 0 else float("nan")
                )

            jain = bdf["jains_index"].dropna() if "jains_index" in bdf.columns else pd.Series(dtype=float)
            rows.append({
                "Method": f"Baseline: {method}",
                "Mean Reward": f"{rewards.mean():.3f} +/- {rewards.std(ddof=1):.3f}" if len(rewards) > 1 else f"{rewards.mean():.3f}",
                "Loss Rate %": f"{loss_rate_pct:.4f}%" if total_arrived > 0 else "N/A",
                "Jain's Index": f"{jain.mean():.4f}" if not jain.empty else "N/A",
                "Type": "Static baseline",
            })

    if not learner_df.empty:
        max_ep = learner_df["episode"].max()
        final_learner = learner_df[learner_df["episode"] == max_ep]
        for method, ldf in final_learner.groupby("method"):
            rewards = ldf["mean_reward"].dropna()
            rows.append({
                "Method": f"Learner: {method} (ep {max_ep})",
                "Mean Reward": f"{rewards.mean():.3f} +/- {rewards.std(ddof=1):.3f}" if len(rewards) > 1 else f"{rewards.mean():.3f}",
                "Loss Rate %": "Logged per step",
                "Jain's Index": "Dynamic policy",
                "Type": "Continual DRL",
            })

    return pd.DataFrame(rows)


def df_to_markdown(df):
    """Convert pandas DataFrame to markdown format without external tabulate dependency."""
    if df.empty:
        return "(empty table)\n"
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        cells = [str(row[c]) if pd.notna(row[c]) else "" for c in df.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate summary tables from WFQ experiment results")
    parser.add_argument("--results-dir", type=Path, default=Path("results/final"),
                        help="Folder containing experiment CSVs (default results/final)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"),
                        help="Folder to save generated tables (default results/tables)")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]

    print(f"Loading results from: {results_dir}")
    print(f"Writing tables to:   {output_dir}")

    learner_df = load_learner_results(results_dir)
    evals_df = load_switch_evals(results_dir)
    baselines_df = load_baselines(results_dir, project_root)

    # 1. Stationary Table
    t1 = build_stationary_table(learner_df)
    if not t1.empty:
        t1.to_csv(output_dir / "table1_stationary.csv", index=False)
        (output_dir / "table1_stationary.md").write_text(df_to_markdown(t1), encoding="utf-8")
        print("\n=== Table 1: Stationary Learning Progression ===")
        print(df_to_markdown(t1))
    else:
        print("\nTable 1 skipped: no stationary learner CSVs found in", results_dir)

    # 2. Switch Forgetting Table
    t2 = build_switch_table(evals_df)
    if not t2.empty:
        t2.to_csv(output_dir / "table2_switch_forgetting.csv", index=False)
        (output_dir / "table2_switch_forgetting.md").write_text(df_to_markdown(t2), encoding="utf-8")
        print("\n=== Table 2: Traffic Switch Forgetting Comparison ===")
        print(df_to_markdown(t2))
    else:
        print("\nTable 2 skipped: no switch evals CSVs found in", results_dir)

    # 3. Baselines Comparison Table
    t3 = build_baselines_table(baselines_df, learner_df)
    if not t3.empty:
        t3.to_csv(output_dir / "table3_baselines.csv", index=False)
        (output_dir / "table3_baselines.md").write_text(df_to_markdown(t3), encoding="utf-8")
        print("\n=== Table 3: Baselines vs Learner Comparison ===")
        print(df_to_markdown(t3))

    print(f"\nAll tables saved in {output_dir}")


if __name__ == "__main__":
    main()
