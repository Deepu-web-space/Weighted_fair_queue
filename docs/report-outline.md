# Report outline

A checklist to write against, not text to copy. Every number you need is in
`docs/results/README.md` or in the paper's own Tables 1 and 2. Two rules for the whole
report: say which summary each number is (an exact episode, or the last-50 average),
and give every quoted number a source, either `docs/results/` or the paper.

Page limit: the plan says 6 to 10 pages and an earlier instruction said 10 to 12.
Settle this with the teacher before the layout is fixed.

## 1. Introduction

- The problem: one link, eight queues, demand that differs by more than six times
  between the busiest and quietest queue, so a fixed equal share is not fair.
- Scale of the problem from our own baseline: the equal split loses 958,216 packets of
  8,298,308 arrived, 11.53%, and the loss concentrates on the busiest queues, Q3
  606,244, Q2 219,381 and Q6 132,562, with none on the four quietest.
- What the paper does and what this project adds, in two sentences each.

## 2. The paper under study, and the gaps we targeted

- Its method: SAC allocating weights across queues, with EWC using a Fisher matrix
  refreshed every 30 episodes.
- Its reported results: SAC -8.744 (sd 1.299), SAC with EWC -2.310 (sd 0.236), and a
  stated EWC gain of 76%. That ratio, computed from the same table, is 70.7% to 73.6%.
- The five gaps we set out to close: one seed only; no plain static WFQ baseline; no
  measurement of forgetting after a change; the offered loads are not matched between
  phases; and the improvement is claimed without a variable-policy comparison.
- One sentence on the improvement we chose: task-anchored EWC, with the anchor frozen
  at the end of the first phase instead of refreshed every 30 episodes.

## 3. Method

- Environment and reward: equation (1), with the symbol values, the division by N, and
  the time basis of the decision interval. These are Q1 to Q3 in
  `docs/paper_equations.md`; they must be answered before this section can be written,
  and any assumption we make in place of a paper value must be labelled as ours.
- SAC: the settings table from `config/base.yaml`.
- EWC: the penalty, lambda 2271 for eight queues, Fisher estimated from 5 mini-batches,
  refreshed every 30 episodes in the control run.
- Task-anchored EWC: the anchor is frozen at the end of phase A. State the measure the
  change is judged by: reward on the first traffic pattern, before against after the
  switch.
- The switch protocol: 500 episodes on traffic A, 500 on traffic B, where B is A with
  the queue order reversed, same total load, so the comparison is not confounded by
  load. Five deterministic evaluation episodes per checkpoint, same seeds for every
  method.

## 4. Experimental setup

- Seeds 0 to 4 for reporting, tuning seeds 100 to 104 used only for checking.
- 1,000 episodes of 2,000 simulated seconds per run, 100 decisions per episode.
- 25 jobs, 12.5 hours of wall time at two jobs at a time on one laptop, CPU only. The
  longest single job was 12,040 seconds.
- That nothing changed after the run: `run_info.txt` records commit `d183fd8`, the
  launcher at that commit is byte-identical to the launcher in the repository, and the
  config copy taken at launch is byte-identical to `config/base.yaml`. Report this,
  because it is what makes the numbers attributable to code the reader can inspect.

## 5. Results

- Stationary learning: Table 1 and figure 1. Both methods drift down across 1,000
  episodes. Over the last 50 episodes plain SAC ends at -27.769 and SAC with EWC at
  -27.540, paired difference +0.229, t = 0.22. Say plainly that the published gain does
  not reproduce in five seeds.
- Where fairness fails: the baseline numbers above, plus Jain's index 0.9873 for the
  equal split against 1.0000 for the demand-proportional split. Explain that Jain's
  index in this setup looks good even when 11.53% of packets are lost, so it cannot be
  the only fairness measure.
- The learner against the static baselines: the equal split at -11.655 against the
  learner near -27.4. The learner never beats a fixed split in this setup.
- Forgetting: Table 2 and figures 2 and 3. Drops of 4.438 for plain SAC, 1.491 for SAC
  with EWC and 5.690 for task-anchored, with the paired tests. EWC retaining best is
  suggestive at p about 0.10; the anchored variant does not win.
- Critic loss: 98,484 against 2,429 at episode 1,000, and 100,853 against 1,129 at
  episode 300. Present it as an observation with an open cause, not as an explanation.

## 6. Discussion

- What reproduced: the pipeline, the baselines, the general shape of learning, and the
  fact that EWC changes retention.
- What did not: the paper's reward gain, the direction of our own improvement, and any
  claim that a learned policy beats static WFQ here.
- The reward-scale mismatch: our per-episode reward sits near -27 while the paper
  reports near -8.7, and the paper does not give the holding and loss coefficients, so
  the two numbers are not comparable. This is the most important caveat in the report.
- Why it might have gone this way, stated as hypotheses: the reward scale and the
  unknown coefficients change what the EWC penalty fights against; the critic loss
  under EWC grows by about 40 times, which can destabilise the value estimates; and
  500 episodes per phase may be too short for a retention advantage to show at five
  seeds.

## 7. Limitations

- The Block 4 pilot did not meet its stop rule (block means -19.511, -20.052, -16.739,
  -25.088) and gate item 1 was waived, with the reason in the decision log. Report the
  failure rather than the waiver alone.
- Learner fairness metrics cannot be computed: the learner CSVs do not log per-decision
  allocations, so MSE, MAE, RMSE and a learner Jain's index are unavailable.
- Five seeds, one machine, one pair of traffic patterns, one queue count.
- The critic-loss difference has no established cause.
- Anything the paper states that we could not verify because the PDF equation is an
  image: say so where it matters.

## 8. Conclusion

- Three or four sentences: the setup was rebuilt and verified; the published reward
  gain does not reproduce here; the anchored variant does not beat plain SAC on
  retention; the learned policy does not beat a static split; and the next measurement
  that would separate the two effects, a control run that trains on traffic A for all
  1,000 episodes with no switch, so the drift both curves show can be told apart from
  forgetting.

## 9. References and appendix

- The paper under study, plus the SAC and EWC sources, plus anything you cite for the
  baselines.
- Appendix: how to reproduce (the commands in the README), the full decision log, and
  the test counts (79 tests, `ALL CHECKS PASSED` for the baseline check).
