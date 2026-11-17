# Weighted Fair Queueing with Continual Deep Reinforcement Learning

Reproduction and extension of a published continual deep-RL study on weighted fair
queueing (WFQ). A single Soft Actor-Critic (SAC) agent allocates bandwidth across
eight queues, then the traffic pattern changes and the agent has to keep serving the
old pattern well while adapting to the new one. The published method uses Elastic
Weight Consolidation (EWC) with a Fisher matrix refreshed every 30 episodes. This
project reproduces that method and tests one change to it: a task-anchored EWC, where
the anchor is frozen once at the end of the first traffic phase instead of being
refreshed.

The paper is MDPI *Telecom* 2025, 6(3), 46, open access under CC BY. Paper values
quoted here come from its Tables 1 and 2 and its Section 5.

## Status

- Final five-seed run complete: 25 jobs, 40 CSV files, `failed: 0`, 750.5 minutes of
  wall time, launched 16 Nov 2026 18:53 from commit `d183fd8`.
- Tables and figures for the report are generated and stored in `docs/results/`.
- Not yet complete: the decision log rewrite, the Q1 to Q3 paper answers, and the
  reported reasons for the Block 4 waiver. See `docs/decisions.md`.

## What the results show, in one paragraph

Both methods learn something and then get worse over the 1,000 episodes, and the
learned policy ends far below a fixed equal split of bandwidth. Across five seeds the
published EWC gain does not appear: over the last 50 episodes plain SAC ends at
-27.769 and SAC with EWC at -27.540, a paired difference of +0.229 (t = 0.22). In the
traffic-switch test SAC with EWC forgets 2.95 units less than plain SAC (paired
t = -2.10, p about 0.10), while the task-anchored variant this project proposes
forgets 1.25 units more than plain SAC and ends lowest on the new traffic. The
anchored EWC critic loss also grows about 40 times larger than plain SAC's. The
results are reported as they came out, including the parts that do not support the
method under test.

## Repository layout

| Path | What it holds |
|---|---|
| `src/wfq/` | Library: `simulator.py`, `env.py`, `networks.py`, `replay.py`, `sac.py`, `ewc.py`, `train.py`, `switch.py`, `baselines.py`, `metrics.py` |
| `config/base.yaml` | Every tunable value. No number goes in the code; each value is read from here |
| `experiments/` | Runners: `run_baselines.py`, `run_learner.py`, `run_switch.py`, `run_block7.py` (all final jobs), plus `check_baselines.py` and `check_learning.py` |
| `analysis/` | `make_tables.py` and `make_plots.py`, the report tables and figures |
| `docs/` | `decisions.md`, `paper_equations.md`, and `results/` with the final tables and figures |
| `tests/` | 12 test modules, run with pytest |
| `results/` | Local run output. Git-ignored except three baseline CSV files kept as evidence |

## Setup

Recorded environment: Windows, Python 3.10.11, torch 2.6.0, CPU only, matplotlib for
figures. For a CPU-only machine:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy pyyaml matplotlib pandas pytest
python -m pytest -q
```

The suite reports `79 passed` on the last full run. Use the virtual environment for
every command below; on a machine whose global Python has no pytest, `python -m pytest`
fails with `No module named pytest`.

## How to run

**Baselines** (equal split and demand-proportional split, five seeds from the config):

```powershell
python experiments\run_baselines.py
python experiments\check_baselines.py        # ends with: ALL CHECKS PASSED
```

Output lands in `results\` as `baselines_equal.csv` and `baselines_proportional.csv`.
These two files are tracked in git; check `git diff` before committing anything that
overwrites them.

**One learner run** (for a single check; the full sweep is below):

```powershell
python experiments\run_learner.py --method sac --seed 0
python experiments\run_learner.py --method sac_ewc --seed 0
python experiments\check_learning.py results\sac_seed0.csv
```

**One traffic-switch run** (phase A traffic for 500 episodes, then phase B):

```powershell
python experiments\run_switch.py --method sac --seed 0
python experiments\run_switch.py --method sac_ewc --seed 0
python experiments\run_switch.py --method task_anchored --seed 0
```

**All final jobs** (10 learner runs and 15 switch runs, seeds 0 to 4):

```powershell
python experiments\run_block7.py --dry-run               # lists 25 jobs, starts nothing
python experiments\run_block7.py --gate-passed --parallel 2
python experiments\run_block7.py --gate-passed --parallel 2 --resume
```

Notes on `run_block7.py`:

- It refuses to start without `--gate-passed`. That flag is an attestation typed by
  the person running it, not a check the code performs.
- Without `--resume` it refuses to overwrite any existing output file. With `--resume`
  a job is treated as done only if all of its files exist **and** pass a validator
  (row counts, numeric sanity). A partial file from a killed job must still be deleted
  by hand before a rerun.
- The full sweep took 12.5 hours at `--parallel 2` on the laptop (CPU, two threads per
  job from `training.torch_threads`). The longest single job took 12,040 seconds.
- Every run is 1,000 episodes of 2,000 simulated seconds, with the agent acting every
  20 seconds, so 100 decisions per episode.

**Tables and figures** for the report:

```powershell
python analysis\make_tables.py --results-dir results\final --output-dir results\tables
python analysis\make_plots.py  --results-dir results\final --output-dir results\figures
```

**Checks:**

```powershell
python -m pytest -q                          # 79 passed
python experiments\check_baselines.py         # ALL CHECKS PASSED
python experiments\check_learning.py results\final\sac_seed0.csv
git check-ignore -v results\final\sac_seed0.csv    # prints the ignore rule
```

## Provenance of the final run

`results/final/run_info.txt` records the launch:

```
started: 2026-11-16T18:53:37
commit: d183fd8
mode: full
parallel: 2
python: 3.10.11
jobs: 25
```

`results/final/summary.txt` records one line per job with its wall time and exit code,
21 jobs run and 4 skipped as already complete from an earlier launch, and ends with
`failed: 0`. The launcher at that commit is byte-identical to the launcher now in the
repository, and `results/final/base_config_used.yaml` is byte-identical to
`config/base.yaml`, so the settings quoted in this README are the settings the run
used.

Disclosure: the batch was started from commit `d183fd8` after `--gate-passed` was
typed while gate items 2 to 5 were still open. That is recorded here rather than left
implied.

The raw run files under `results/final/` are deliberately not tracked by git. They are
kept on the machine that ran them, and the tables, figures and summary that matter are
committed under `docs/results/`.

## Known issues and open items

1. `docs/results/table3_baselines.md` prints the loss rate 100 times too small: it
   shows `0.115%` where the value is `11.53%`. Fix the format in
   `analysis/make_tables.py` (`build_baselines_table`, the `Loss Rate %` line), rerun,
   and commit the corrected table.
2. The same table's two learner rows carry the placeholders "Logged per step" and
   "Dynamic policy" in the Loss Rate and Jain's Index columns. The learner CSVs do not
   log per-decision allocations, so those two quantities cannot be computed from the
   current results. State that, or leave the learner rows out of the report table.
3. The `" +/- "` figure in Table 3 is the spread of single episodes pooled across
   5 seeds and 1,000 episodes, not the spread across seeds. The across-seed standard
   deviation of the equal split is 0.033. Label it correctly in the report.
4. `docs/decisions.md` is still the six-row stub from Day 3 and does not match the
   decision letters used in `config/base.yaml`. See `docs/decisions-draft.md`.
5. `docs/paper_equations.md` has three unanswered questions, Q1 to Q3, marked
   "To be verified". They concern equation (1): the symbol values, the division by N,
   and the time basis of the decision interval.
6. The Block 4 pilot run (one seed, 200 episodes) did not pass its own stop rule: the
   block means went -19.511, -20.052, -16.739 (best) and -25.088, so the average never
   climbed and levelled off. Gate item 1 was waived, and the reported reason is still
   missing from the decision log.
7. The published 76% EWC gain does not recompute from the paper's own Table 2
   (the ratio gives 70.7% to 73.6%), and it does not appear in this project's five-seed
   run. Both statements belong in the report as findings.
