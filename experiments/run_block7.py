"""Block 7: run all final experiments (seeds 0 to 4), a few jobs at a time.

Jobs (25 in total):
  run_learner.py  sac, sac_ewc                    x 5 seeds = 10 runs
  run_switch.py   sac, sac_ewc, task_anchored     x 5 seeds = 15 runs

Safety rules built in:
  - Does nothing unless --gate-passed is given (Block 7 gate met).
  - --dry-run prints the plan and runs nothing.
  - Refuses to start if the output folder already holds results, unless --resume.
  - Refuses a full run if git has uncommitted changes (results must match a commit).
  - --quick is a pipeline test with tiny runs. It must never write to results/final.

Each job writes its own log in <output>/logs/. A summary is written to
<output>/summary.txt, and run details to <output>/run_info.txt.
"""

import argparse
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "final"
LEARNER_METHODS = ["sac", "sac_ewc"]
SWITCH_METHODS = ["sac", "sac_ewc", "task_anchored"]

# Pipeline test only (--quick): tiny runs that exercise every code path.
QUICK_LEARNER_EPISODES = 3
QUICK_PHASE_A = 12   # must pass warm-up (about 10 episodes) so task_anchored refreshes
QUICK_PHASE_B = 2


class Job:
    def __init__(self, name, cmd, outputs):
        self.name = name
        self.cmd = cmd
        self.outputs = outputs  # list of Path

    def done(self):
        return all(p.exists() for p in self.outputs)


def build_jobs(cfg, output, quick):
    seeds = cfg["training"]["seeds_final"]
    learner = PROJECT_ROOT / "experiments" / "run_learner.py"
    switch = PROJECT_ROOT / "experiments" / "run_switch.py"
    config_path = PROJECT_ROOT / "config" / "base.yaml"
    jobs = []

    for seed in seeds:
        for method in LEARNER_METHODS:
            cmd = [sys.executable, str(learner), "--method", method,
                   "--seed", str(seed), "--output", str(output)]
            if quick:
                cmd += ["--episodes", str(QUICK_LEARNER_EPISODES)]
            outputs = [output / f"{method}_seed{seed}.csv"]
            jobs.append(Job(f"learner_{method}_seed{seed}", cmd, outputs))

    for seed in seeds:
        for method in SWITCH_METHODS:
            cmd = [sys.executable, str(switch), "--method", method,
                   "--seed", str(seed), "--output", str(output)]
            if quick:
                cmd += ["--phase-a", str(QUICK_PHASE_A), "--phase-b", str(QUICK_PHASE_B)]
            outputs = [output / f"{method}_seed{seed}_episodes.csv",
                       output / f"{method}_seed{seed}_evals.csv"]
            jobs.append(Job(f"switch_{method}_seed{seed}", cmd, outputs))

    return jobs


def git_value(args):
    try:
        out = subprocess.run(["git"] + args, cwd=PROJECT_ROOT, capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_job(job, logs_dir, resume):
    if resume and job.done():
        return job, "skipped", 0.0, 0
    log_path = logs_dir / f"{job.name}.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(job.cmd) + "\n")
        log.flush()
        proc = subprocess.run(job.cmd, cwd=PROJECT_ROOT, stdout=log,
                              stderr=subprocess.STDOUT)
    seconds = time.perf_counter() - started
    status = "ok" if proc.returncode == 0 and job.done() else "FAILED"
    if status == "FAILED":
        # Runners write CSVs row by row, so a crash leaves a partial file.
        # Delete it so --resume reruns this job instead of skipping it.
        for path in job.outputs:
            path.unlink(missing_ok=True)
    return job, status, seconds, proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Block 7: run all final experiments")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output folder (default results/final)")
    parser.add_argument("--parallel", type=int, default=3,
                        help="Jobs at once (default 3; each job uses 2 CPU threads)")
    parser.add_argument("--gate-passed", action="store_true",
                        help="Required for a real run. Only add when the Block 7 gate is met.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and run nothing")
    parser.add_argument("--resume", action="store_true",
                        help="Skip jobs whose output files already exist")
    parser.add_argument("--quick", action="store_true",
                        help="Pipeline test with tiny runs. Never for results.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "base.yaml")
    args = parser.parse_args()

    output = args.output.resolve()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    final = set(cfg["training"]["seeds_final"])
    tuning = set(cfg["training"]["seeds_tuning"])
    if final & tuning:
        sys.exit(f"Refused: final seeds {sorted(final)} overlap tuning seeds {sorted(tuning)}.")

    if args.quick and output == DEFAULT_OUTPUT.resolve():
        sys.exit("Refused: --quick is a pipeline test. Use --output results\\pipeline_test.")

    jobs = build_jobs(cfg, output, args.quick)
    print(f"Plan: {len(jobs)} jobs, seeds {sorted(final)}, output {output}")
    print(f"Mode: {'QUICK (pipeline test)' if args.quick else 'FULL'}, "
          f"{args.parallel} at a time, resume={args.resume}")
    for job in jobs:
        print(f"  {job.name:<32} {' '.join(job.cmd[2:])}")

    if args.dry_run:
        print("Dry run: nothing was started.")
        return

    if not args.gate_passed:
        sys.exit("Not started: add --gate-passed only when the Block 7 gate is met.")

    existing = [j.name for j in jobs if j.done()]
    if existing and not args.resume:
        sys.exit(f"Refused: {len(existing)} result files already exist in {output}. "
                 "Move that folder aside, or add --resume.")

    if not args.quick:
        if git_value(["rev-parse", "--is-inside-work-tree"]) != "true":
            sys.exit("Refused: this folder is not a git repository.")
        if git_value(["status", "--porcelain", "--untracked-files=no"]):
            sys.exit("Refused: git has uncommitted changes. Commit or clean them first "
                     "(git status).")
        print("Running tests before the full run (expect 76 passed)...")
        tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True)
        print(tests.stdout.strip().splitlines()[-1] if tests.stdout.strip() else "no output")
        if tests.returncode != 0:
            sys.exit("Refused: tests failed. Run python -m pytest -q to see why.")

    logs_dir = output / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output / "base_config_used.yaml")
    commit = git_value(["rev-parse", "--short", "HEAD"]) or "unknown"
    (output / "run_info.txt").write_text(
        f"started: {datetime.now().isoformat(timespec='seconds')}\n"
        f"commit: {commit}\n"
        f"mode: {'quick' if args.quick else 'full'}\n"
        f"parallel: {args.parallel}\n"
        f"python: {sys.version.split()[0]}\n"
        f"jobs: {len(jobs)}\n",
        encoding="utf-8")

    results = []
    overall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(run_job, job, logs_dir, args.resume) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            job, status, seconds, code = future.result()
            results.append((job, status, seconds, code))
            print(f"[{count:>2}/{len(jobs)}] {status:<8} {job.name:<32} {seconds:8.1f} s")

    failed = [r for r in results if r[1] == "FAILED"]
    lines = [f"{job.name:<32} {status:<8} {seconds:8.1f} s  exit {code}"
             for job, status, seconds, code in sorted(results, key=lambda r: r[0].name)]
    lines.append(f"total wall time: {(time.perf_counter() - overall) / 60:.1f} min")
    lines.append(f"failed: {len(failed)}")
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-2:]))
    print(f"Summary: {output / 'summary.txt'}")
    if failed:
        print("Some jobs failed. See the logs in", logs_dir)
        sys.exit(1)


if __name__ == "__main__":
    main()
