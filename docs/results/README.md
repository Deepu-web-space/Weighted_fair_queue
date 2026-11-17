# Final results

Three tables and three figures from the final five-seed run, and the definitions and
checks needed to read them. The raw run files under `results/final/` are not tracked by
git; they stay on the machine that ran them.

| File | What it is |
|---|---|
| `table1_stationary.md` / `.csv` | Stationary learning: exact-episode checkpoints over 1,000 episodes, five seeds |
| `table2_switch_forgetting.md` / `.csv` | Traffic-switch test: three deterministic evaluation checkpoints and the forgetting drop, five seeds |
| `table3_baselines.md` / `.csv` | The two static baselines against both learners |
| `fig1_learning_curves.png` | Learning curves, plain SAC against SAC with EWC, mean over five seeds |
| `fig2_traffic_switch.png` | Switch test as grouped bars at the three checkpoints |
| `fig3_switch_trajectories.png` | Training trajectories across the switch, all three methods |

## How the run was set up

Reporting seeds 0 to 4. Tuning seeds 100 to 104 were used only for checking and never
for reported results. Every learner run is 1,000 episodes of 2,000 simulated seconds,
the agent acting every 20 seconds, so 100 decisions per episode. The switch test trains
500 episodes on traffic A, then 500 on traffic B, where traffic B is traffic A with the
queue order reversed and the same total load (0.83 packets per second).

## Definitions, so the numbers cannot be misread

**Table 1.** For each method and each checkpoint episode (50, 100, 200, 300, 400, 500,
800, 1,000), the mean over the five seeds of that episode's mean reward, with the
standard deviation across seeds (ddof = 1). The loss columns are the same average over
seeds. A single episode is a noisy sample: the last-50 figures below are the steadier
summary, and the report should say which one it is quoting at each point.

**Table 2.** Five deterministic evaluation episodes per checkpoint, averaged within a
seed, then the mean and standard deviation across the five seeds. "Before Switch (A)"
is evaluated at the end of phase A on traffic A. "After Switch (A retained)" is
evaluated at the end of phase B on traffic A. "Forgetting Drop (A)" is the first minus
the second, so positive means the method got worse on the old traffic. "After Switch
(B adapted)" is evaluated at the end of phase B on traffic B.

**Table 3.** The baseline rows are pooled over all episodes and seeds, 5 seeds times
1,000 episodes, so the `+/-` figure is the spread of single episodes, **not** the spread
across seeds. The across-seed standard deviation of the equal split is 0.033. The
learner rows use the episode-1,000 checkpoint only. Label the two kinds of spread
differently, or the table contradicts the text.

## Key numbers

Last-50-episode mean reward, taken over episodes 951 to 1,000 per seed and then
averaged across the five seeds, with the standard deviation across seeds:

| Measure | Plain SAC | SAC + EWC | Task-anchored |
|---|---|---|---|
| Last 50 episodes | -27.769 (sd 2.176) | -27.540 (sd 3.507) | not run |
| Episode 1,000 (Table 1) | -27.421 +/- 3.438 | -29.984 +/- 7.252 | not run |
| Forgetting drop (Table 2) | 4.438 +/- 4.462 | 1.491 +/- 3.159 | 5.690 +/- 2.629 |
| End of phase B on traffic B | -27.062 +/- 1.791 | -25.072 +/- 5.217 | -28.992 +/- 5.809 |
| Critic loss at episode 1,000 | 2,429 | 98,484 | not run |

Paired comparisons across the five seeds, computed from the raw CSVs and not stored in
this folder:

- Stationary, EWC minus SAC on the last 50 episodes: +0.229, t = 0.22.
- First 50 to last 50 episodes: plain SAC -8.08, SAC with EWC -6.97. Both drift down.
- Forgetting drop, EWC minus SAC: -2.947 (t = -2.10, p about 0.10).
- Forgetting drop, task-anchored minus SAC: +1.252 (t = 0.43).
- Forgetting drop, task-anchored minus EWC: +4.199 (t = 1.78, p about 0.15).
- Critic loss, EWC minus SAC: 98,484 against 2,429 at episode 1,000, and 100,853
  against 1,129 at episode 300.
- With five seeds and p about 0.10, the EWC retention advantage is suggestive, not
  established. Say it that way.

Baselines, over 5 seeds and 1,000 episodes each:

| Baseline | Mean reward | Loss rate | Jain's index | Across-seed sd |
|---|---|---|---|---|
| Equal split | -11.655 | 11.53% of arrived packets | 0.9873 | 0.033 |
| Demand-proportional | -3.467 | 0.005% | 1.0000 | 0.007 |

The equal split loses 958,216 packets of 8,298,308 arrived, and the loss sits on the
busiest queues: Q3 606,244, Q2 219,381, Q6 132,562, Q4 29, and none on Q1, Q5, Q7 and
Q8. That is the concrete version of "equal is not fair when demand differs".

For comparison with the published work: the paper reports SAC at -8.744 (sd 1.299) and
SAC with EWC at -2.310 (sd 0.236), and describes the EWC gain as 76%. Its own table
gives a ratio of 70.7% to 73.6%. Our reward has a different scale (near -27 against
near -8.7 per episode), so the two numbers cannot be compared directly, which belongs
in the discussion rather than being smoothed over.

## Reading the figures

- `fig1_learning_curves.png`: per-episode reward, smoothed, averaged over five seeds
  with the spread shaded. Both curves trend down after the early episodes.
- `fig2_traffic_switch.png`: the three checkpoints as grouped bars, standard deviation
  across seeds as the error bars. Bars show levels, not the size of the drop, so put
  the drop values in the caption.
- `fig3_switch_trajectories.png`: training reward over all 1,000 episodes with the
  switch at episode 500. Plain SAC and task-anchored are numerically identical in
  phase A by design, since the anchored method carries no penalty before the switch,
  so the two lines coincide and the green one hides the blue one. Without that note
  the figure looks as if the anchored method was ahead before the switch, and it was
  not. I verified this from the run files: the largest difference across phase A is 0.0.

## Regenerating

```powershell
python analysis\make_tables.py --results-dir results\final --output-dir results\tables
python analysis\make_plots.py  --results-dir results\final --output-dir results\figures
```

Both scripts also run their own checks under pytest:

```powershell
python -m pytest tests\test_analysis.py -q     # 3 passed
```

Then copy the output over this folder and commit only these files, not `results/`.

## Defects to fix before the report is final

1. `table3_baselines.md` prints the loss rate 100 times too small: `0.115%` should read
   `11.53%`. In `analysis/make_tables.py`, `build_baselines_table`, the `Loss Rate %`
   line multiplies nothing by 100. Fix, rerun, commit.
2. The learner rows in the same table carry "Logged per step" and "Dynamic policy" in
   the Loss Rate and Jain's Index columns. The learner CSVs log no per-decision
   allocations, so those two quantities cannot be computed from the current results.
   Either state that in the table's caption, or keep the two learner rows out of the
   report table and cite the baseline rows only.
3. The `+/-` figures in Table 3 are pooled per-episode spread, not across-seed spread.
   See the definitions above.

## Provenance

`results/final/run_info.txt`: started 2026-11-16T18:53:37, commit `d183fd8`, full mode,
parallel 2, Python 3.10.11, 25 jobs. `results/final/summary.txt`: 21 jobs run, 4 skipped
as already complete, every exit code 0, total wall time 750.5 minutes, `failed: 0`.
`results/final/base_config_used.yaml` is byte-identical to `config/base.yaml`, so the
settings quoted in the report are the settings the run used.
