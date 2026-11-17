# Decision log: fill-in draft

The current `docs/decisions.md` is the six-row stub from Day 3. It does not match the
decision letters used in `config/base.yaml`, where queue capacity is H, warm-up is B,
the Fisher batch count is D, the phase B traffic is J, and the Dirichlet floor is
called an "A detail". Use this file to replace it: the Choice column is already filled
from the config and the code, so what is missing is the reasoning, in your words, one
or two sentences per row. Then re-letter the rows so the log and the config agree, or
drop the letters and rely on the setting key.

Rule for the finished log: every literal in the code that is not read from the config
must appear here, with a line saying where it lives.

## 1. Environment and simulator

| ID | Decision | Choice made, from the config or the code | Why (write this) |
|---|---|---|---|
| H | Queue capacity | 32 packets per queue |  |
|  | Decision interval | Agent acts every 20 simulated seconds |  |
|  | Episode length | 2,000 simulated seconds, so 100 decisions per episode |  |
|  | Arrival process | Gamma interarrival and service times, squared coefficient of variation 0.8 |  |
|  | Arrival rates | The eight-queue set from the paper's first experiment: 0.03, 0.16, 0.20, 0.10, 0.08, 0.15, 0.06, 0.05 packets per second |  |
|  | Total bandwidth | 1.0 packets per second, so total offered load is 0.83 |  |
|  | Random streams | One independent stream per queue |  |
|  | Episode termination | `done` stays False: the simulation is continuous |  |
|  | Weight normalisation | Allocation weights are normalised so they sum to 1 |  |
|  | Weight floor | `min_weight` 1.0e-6, so no queue is starved by an exact zero |  |
|  | State normalisation | Confirm against the paper and state what the code does |  |

## 2. Learning method

| ID | Decision | Choice made | Why (write this) |
|---|---|---|---|
| B | Warm-up | No updates until 1,000 transitions, about 10 episodes |  |
|  | SAC values | gamma 0.99, tau 0.01, alpha 0.2, actor learning rate 1e-4, critic 1e-3, batch 128, replay 10,000, hidden units 64 |  |
|  | Dirichlet floor | `min_concentration` 1.0 added to the actor output so the proposal is never spiky |  |
|  | EWC strength | lambda 2271, the paper's value for 8 queues (its 12-queue value is 2275) |  |
| D | Fisher estimate | 5 mini-batches averaged for the Fisher values |  |
|  | Fisher refresh | Every 30 episodes, from the paper |  |
| I | Task-anchored EWC (our change) | At the end of phase A the weights and Fisher are frozen once as the anchor and never refreshed; the paper's 30-episode refresh stays in the code as the control |  |

## 3. Experimental protocol

| ID | Decision | Choice made | Why (write this) |
|---|---|---|---|
|  | Reporting seeds | 0 to 4, five seeds |  |
|  | Tuning seeds | 100 to 104, never used for a reported number |  |
|  | Episode seeds | `episode_seed_stride` 100,000, so a run's episode seed is run seed times 100,000 plus the episode number |  |
|  | Evaluation seeds | 900,000,000 plus run seed times 100 plus k, for five deterministic evaluation episodes, the same for every method |  |
|  | Checkpoints | Episodes 50, 100, 200, 300, 400, 500, 800 and 1,000 in Table 1, plus the last-50 summary |  |
| J | Phase B traffic | Phase A rates with the queue order reversed, so the total load is unchanged at 0.83 |  |
|  | Phase lengths | 500 episodes on traffic A, then 500 on traffic B |  |
|  | Threads and parallelism | 2 threads per job, 2 jobs at a time on the laptop |  |

## 4. The Block 4 waiver

Gate item 1 asked the Block 4 learning check to pass before the final run. It did not:
the pilot run (seed 100, 200 episodes) gave block means of -19.511, -20.052, -16.739
(the best) and -25.088, with a first-block-to-last-block change of -5.036, so the
average never climbed and levelled off as the stop rule required. No setting was
changed for that run.

Drafted sentence, with the reason left for you:

> Gate item 1, the Block 4 learning check, is waived. The pilot did not meet the stop
> rule, and no setting was changed for it. Reason: ______. Signed: ______, 17 Nov 2026.

## 5. Gate checklist as it stands

| Item | State |
|---|---|
| 1. Block 4 check passes | Waived, reason above to be written |
| 2. Q1 to Q3 answered in your words and committed | Open, still "To be verified" in `docs/paper_equations.md` |
| 3. Settings tidy committed | Open, and now unnecessary: editing `config/base.yaml` after the run only weakens the "nothing changed" record, for no value change |
| 4. Step 6 baselines committed and checked | Checked: `check_baselines.py` ends `ALL CHECKS PASSED` |
| 5. Working tree clean and `results/` ignored | Partly met: the ignore rule is anchored, three baseline CSVs stay tracked on purpose, and `venv/` is still to be removed or ignored |

## 6. Disclosure

The final batch was launched on 16 Nov 2026 from commit `d183fd8` with
`--gate-passed` typed while items 2 to 5 above were still open. That flag is an
attestation by the person running the batch, not a check the code performs. The
attestation is recorded here and in `results/final/run_info.txt` rather than left
implied.
