"""Block 7: validated final experiment runner."""

import argparse
import csv
import math
import os
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

QUICK_LEARNER_EPISODES = 3
QUICK_PHASE_A = 12
QUICK_PHASE_B = 2

LEARNER_COLUMNS = [
    "episode", "mean_reward", "updates",
    "critic_loss", "actor_loss",
    "critic_penalty", "actor_penalty",
]

SWITCH_EPISODE_COLUMNS = [
    "episode", "phase", "mean_reward", "updates",
    "critic_loss", "actor_loss",
    "critic_penalty", "actor_penalty",
]

EVAL_COLUMNS = [
    "checkpoint", "traffic", "eval_index", "mean_reward"
]


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Missing CSV header")
        return reader.fieldnames, list(reader)


def numeric(value, allow_nan=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False

    if allow_nan and math.isnan(number):
        return True

    return math.isfinite(number)


def validate_learner(path, expected_episodes):
    try:
        headers, rows = read_csv(path)

        if headers != LEARNER_COLUMNS:
            return False

        if len(rows) != expected_episodes:
            return False

        for index, row in enumerate(rows, start=1):
            if int(row["episode"]) != index:
                return False

            if not numeric(row["mean_reward"]):
                return False

            if not numeric(row["updates"]):
                return False

            for key in (
                "critic_loss", "actor_loss",
                "critic_penalty", "actor_penalty"
            ):
                if not numeric(row[key], allow_nan=True):
                    return False

        return True

    except (OSError, ValueError, TypeError, KeyError):
        return False


def validate_switch_episodes(path, phase_a, phase_b):
    try:
        headers, rows = read_csv(path)
        total = phase_a + phase_b

        if headers != SWITCH_EPISODE_COLUMNS:
            return False

        if len(rows) != total:
            return False

        for index, row in enumerate(rows, start=1):
            if int(row["episode"]) != index:
                return False

            expected_phase = "A" if index <= phase_a else "B"

            if row["phase"] != expected_phase:
                return False

            if not numeric(row["mean_reward"]):
                return False

            if not numeric(row["updates"]):
                return False

            for key in (
                "critic_loss", "actor_loss",
                "critic_penalty", "actor_penalty"
            ):
                if not numeric(row[key], allow_nan=True):
                    return False

        return True

    except (OSError, ValueError, TypeError, KeyError):
        return False


def validate_switch_evals(path, eval_episodes):
    try:
        headers, rows = read_csv(path)

        if headers != EVAL_COLUMNS:
            return False

        expected = {
            ("end_A", "A"),
            ("end_B", "A"),
            ("end_B", "B"),
        }

        if len(rows) != 3 * eval_episodes:
            return False

        observed = {}

        for row in rows:
            checkpoint = row["checkpoint"]
            traffic = row["traffic"]
            index = int(row["eval_index"])

            if (checkpoint, traffic) not in expected:
                return False

            if index < 0 or index >= eval_episodes:
                return False

            if not numeric(row["mean_reward"]):
                return False

            key = (checkpoint, traffic)
            observed.setdefault(key, []).append(index)

        if set(observed) != expected:
            return False

        for indices in observed.values():
            if sorted(indices) != list(range(eval_episodes)):
                return False

        return True

    except (OSError, ValueError, TypeError, KeyError):
        return False


class Job:
    def __init__(self, name, cmd, outputs, validator):
        self.name = name
        self.cmd = cmd
        self.outputs = outputs
        self.validator = validator

    def done(self):
        return self.validator()


def build_jobs(cfg, output, quick):
    seeds = cfg["training"]["seeds_final"]

    learner = PROJECT_ROOT / "experiments" / "run_learner.py"
    switch = PROJECT_ROOT / "experiments" / "run_switch.py"

    phase_a = QUICK_PHASE_A if quick else cfg["switch_test"]["phase_a_episodes"]
    phase_b = QUICK_PHASE_B if quick else cfg["switch_test"]["phase_b_episodes"]

    learner_episodes = (
        QUICK_LEARNER_EPISODES if quick else cfg["training"]["episodes"]
    )

    eval_episodes = cfg["switch_test"]["eval_episodes"]

    jobs = []

    for seed in seeds:
        for method in LEARNER_METHODS:
            cmd = [
                sys.executable, str(learner),
                "--method", method,
                "--seed", str(seed),
                "--output", str(output),
            ]

            if quick:
                cmd += ["--episodes", str(QUICK_LEARNER_EPISODES)]

            csv_path = output / f"{method}_seed{seed}.csv"

            jobs.append(Job(
                f"learner_{method}_seed{seed}",
                cmd,
                [csv_path],
                lambda p=csv_path, n=learner_episodes:
                    validate_learner(p, n),
            ))

    for seed in seeds:
        for method in SWITCH_METHODS:
            cmd = [
                sys.executable, str(switch),
                "--method", method,
                "--seed", str(seed),
                "--output", str(output),
            ]

            if quick:
                cmd += [
                    "--phase-a", str(QUICK_PHASE_A),
                    "--phase-b", str(QUICK_PHASE_B),
                ]

            episodes_path = output / f"{method}_seed{seed}_episodes.csv"
            evals_path = output / f"{method}_seed{seed}_evals.csv"

            jobs.append(Job(
                f"switch_{method}_seed{seed}",
                cmd,
                [episodes_path, evals_path],
                lambda ep=episodes_path, ev=evals_path,
                       a=phase_a, b=phase_b, n=eval_episodes:
                    validate_switch_episodes(ep, a, b)
                    and validate_switch_evals(ev, n),
            ))

    return jobs


def git_value(args):
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_job(job, logs_dir, resume):
    if resume and job.done():
        return job, "skipped", 0.0, 0

    # Remove incomplete or stale outputs before restarting the job.
    for path in job.outputs:
        path.unlink(missing_ok=True)

    log_path = logs_dir / f"{job.name}.log"
    started = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(job.cmd) + "\n")
        log.flush()

        proc = subprocess.run(
            job.cmd,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    seconds = time.perf_counter() - started

    status = (
        "ok"
        if proc.returncode == 0 and job.done()
        else "FAILED"
    )

    if status == "FAILED":
        for path in job.outputs:
            path.unlink(missing_ok=True)

    return job, status, seconds, proc.returncode


def acquire_lock(output):
    lock_path = output / ".block7.lock"

    try:
        fd = os.open(
            str(lock_path),
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError:
        sys.exit(
            f"Refused: lock already exists: {lock_path}\n"
            "Check that no Block 7 runner is active. "
            "Only remove the lock manually after confirming "
            "all Block 7, learner and switch processes have stopped."
        )

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"pid={os.getpid()}\n")
        f.write(f"started={datetime.now().isoformat()}\n")

    return lock_path


def main():
    parser = argparse.ArgumentParser(
        description="Block 7: validated final experiment runner"
    )

    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--gate-passed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "base.yaml",
    )

    args = parser.parse_args()

    if args.parallel < 1:
        sys.exit("Refused: --parallel must be at least 1.")

    output = args.output.resolve()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    final = set(cfg["training"]["seeds_final"])
    tuning = set(cfg["training"]["seeds_tuning"])

    if final & tuning:
        sys.exit(
            f"Refused: final seeds {sorted(final)} overlap "
            f"tuning seeds {sorted(tuning)}."
        )

    if args.quick and output == DEFAULT_OUTPUT.resolve():
        sys.exit(
            "Refused: --quick is a pipeline test. "
            "Use results\\pipeline_test."
        )

    jobs = build_jobs(cfg, output, args.quick)

    print(f"Plan: {len(jobs)} jobs, seeds {sorted(final)}, output {output}")
    print(
        f"Mode: {'QUICK' if args.quick else 'FULL'}, "
        f"{args.parallel} at a time, resume={args.resume}"
    )

    for job in jobs:
        print(f"  {job.name:<32} {' '.join(job.cmd[2:])}")

    if args.dry_run:
        print("Dry run: nothing was started.")
        return

    if not args.gate_passed:
        sys.exit(
            "Not started: add --gate-passed only when the Block 7 gate is met."
        )

    # Without --resume, refuse to overwrite any existing output.
    existing_files = [
        path
        for job in jobs
        for path in job.outputs
        if path.exists()
    ]

    if existing_files and not args.resume:
        sys.exit(
            f"Refused: {len(existing_files)} output files already exist "
            f"in {output}. Use --resume after checking the files."
        )

    if not args.quick:
        if git_value(["rev-parse", "--is-inside-work-tree"]) != "true":
            sys.exit("Refused: this folder is not a git repository.")

        if git_value(["status", "--porcelain", "--untracked-files=no"]):
            sys.exit(
                "Refused: git has uncommitted changes. "
                "Commit or clean them first."
            )

        print("Running tests before the full run...")

        tests = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        print(
            tests.stdout.strip().splitlines()[-1]
            if tests.stdout.strip()
            else "no output"
        )

        if tests.returncode != 0:
            sys.exit("Refused: tests failed.")

    output.mkdir(parents=True, exist_ok=True)

    # Prevent two runners from writing into the same output directory.
    lock_path = acquire_lock(output)

    try:
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
            encoding="utf-8",
        )

        # Validate existing files before launching jobs.
        if args.resume:
            print("\nResume validation:")
            for job in jobs:
                print(
                    f"  {'COMPLETE' if job.done() else 'INCOMPLETE':<10} "
                    f"{job.name}"
                )

        results = []
        overall = time.perf_counter()

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = [
                pool.submit(run_job, job, logs_dir, args.resume)
                for job in jobs
            ]

            for count, future in enumerate(as_completed(futures), 1):
                job, status, seconds, code = future.result()
                results.append((job, status, seconds, code))

                print(
                    f"[{count:>2}/{len(jobs)}] "
                    f"{status:<8} {job.name:<32} {seconds:8.1f} s"
                )

        failed = [r for r in results if r[1] == "FAILED"]

        lines = [
            f"{job.name:<32} {status:<8} {seconds:8.1f} s  exit {code}"
            for job, status, seconds, code
            in sorted(results, key=lambda r: r[0].name)
        ]

        lines.append(
            f"total wall time: "
            f"{(time.perf_counter() - overall) / 60:.1f} min"
        )
        lines.append(f"failed: {len(failed)}")

        (output / "summary.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        print("\n".join(lines[-2:]))
        print(f"Summary: {output / 'summary.txt'}")

        if failed:
            print("Some jobs failed. See logs in", logs_dir)
            sys.exit(1)

        # Final independent validation of every output.
        invalid = [job.name for job in jobs if not job.done()]

        if invalid:
            print("Final validation failed:", invalid)
            sys.exit(1)

        print("\nAll experiment outputs passed CSV validation.")

    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()