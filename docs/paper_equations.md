# Paper Equations and Mathematical Definitions

This document records the mathematical definitions in the reference paper and distinguishes them from the implementation in this repository.

**Reference paper**

Mays A. Mawlood and Dhari Ali Mahmood, "Optimizing Weighted Fair Queuing with Deep Reinforcement Learning for Dynamic Bandwidth Allocation," *Telecom*, 2025, 6(3), Article 46. DOI: https://doi.org/10.3390/telecom6030046.

**Purpose:** Provide a mathematical reference for the reward, state, action, SAC, EWC, and evaluation equations. Statements marked *Paper* are taken from the reference. Statements marked *Project implementation* describe this repository. Differences and ambiguities are explicitly identified.

---

## 1. Reward Function (Paper Equation 1)

**Paper location:** Section 4.4, printed page 7. The coefficients and eight-queue form are repeated in Section 5, printed page 17.

### 1.1 General equation

The paper gives:

\[
\boxed{
r(q,a)=-\sum_{i=1}^{N}(h_iq_i+c_il_i)
}
\tag{1}
\]

Where:

- \(r(q,a)\): Reward for queue state \(q\) and action \(a\).
- \(N\): Number of queues.
- \(q_i\): Current number of packets in queue \(i\), including packets in service as described in Section 4.4.
- \(l_i\): Number of packets lost from queue \(i\) in the relevant unit of time.
- \(h_i\): Holding-cost coefficient per packet per unit of time for queue \(i\).
- \(c_i\): Packet-loss penalty coefficient for queue \(i\).

The negative sign means that larger queueing cost and packet loss reduce reward. A value closer to zero is better when comparing runs using the same reward definition and time basis.

### 1.2 Q1: Is the holding cost divided by N?

**Answer: Yes.**

Section 5, printed page 17, explicitly specifies:

\[
\boxed{h_i=\frac{1}{N}}
\]

For the eight-queue experiment:

\[
N=8,\qquad h_i=\frac{1}{8}=0.125
\]

Therefore:

\[
\sum_{i=1}^{N}h_iq_i
=
\frac{1}{N}\sum_{i=1}^{N}q_i
\]

This is a coefficient specified by the paper, not an assumption based on the implementation.

### 1.3 Q2: Is the packet-loss penalty divided by N?

**Answer: No.**

Section 5, printed page 17, specifies:

\[
\boxed{c_i=1}
\]

Therefore:

\[
\sum_{i=1}^{N}c_il_i
=
\sum_{i=1}^{N}l_i
\]

The packet-loss counts are summed directly. There is no division by \(N\) in the loss term.

For eight queues, the complete reward becomes:

\[
\boxed{
r(q,a)=-
\left[
\frac{1}{8}\sum_{i=1}^{8}q_i
+
\sum_{i=1}^{8}l_i
\right]
}
\]

### 1.4 Q3: Is holding cost per second or per 20-second decision?

**Answer: The paper defines the holding-cost coefficient per unit of time, not as a special 20-second coefficient.**

Section 4.4, printed page 7, describes holding cost as the cost per unit of time for each packet in a queue. Section 5.1, printed page 18, separately states that the agent takes an action every 20 seconds.

These statements establish two different things:

- The holding-cost coefficient \(h_i\) is a per-unit-time cost coefficient.
- The agent's control/action interval in Experiment 1 is 20 seconds.

The paper does **not** clearly specify in Equation (1) whether \(q_i\) is integrated over the complete 20-second interval or sampled at one instant. The 20-second action interval alone does not prove that the reward is a time integral.

### 1.5 Project implementation

**Code location:** `src/wfq/env.py`, method `WFQEnv.step()`.

The environment:

1. Records packet-loss and served counters.
2. Applies the normalized action weights.
3. Advances the simulator by `decision_interval_seconds`.
4. Calculates packet losses during that interval from counter differences.
5. Samples queue lengths after the simulator advance.
6. Computes holding cost from that end-of-interval queue snapshot.

Relevant implementation:

```python
self.system.advance(self.decision_interval_seconds)

queue_lengths = [
    q.in_system()
    for q in self.system.queues
]

holding_cost = sum(queue_lengths) / self.n_queues
loss_penalty = sum(lost_counts)
reward = -(holding_cost + loss_penalty)
```

The implemented expression is:

\[
\boxed{
r_t^{\text{code}}=
-\left[
\frac{1}{N}\sum_{i=1}^{N}q_i(t_{\text{end}})
+
\sum_{i=1}^{N}l_i[t,t+\Delta t]
\right]
}
\]

Here:

- \(q_i(t_{\text{end}})\) is the queue length sampled at the end of the decision interval.
- \(l_i[t,t+\Delta t]\) is the number of packets lost during that interval.

**Conclusion:** The coefficient scaling matches the paper (\(h_i=1/N\), \(c_i=1\)). The time measurement does not implement a continuous-time integral of queue length. It uses one end-of-interval snapshot. This is a discrete implementation choice and must be disclosed as a methodological difference.

Do not describe it as an exact time-integrated holding-cost calculation.

---

## 2. State Representation and Normalization

**Paper location:** Section 4.4, printed page 7; Section 4.5, printed page 10.

### 2.1 State definition in the paper

The paper defines the state from the queue lengths:

\[
q=(q_1,q_2,\ldots,q_N)
\]

Each \(q_i\) represents the number of packets either being served or waiting in queue \(i\), with the finite-buffer constraint:

\[
0\le q_i\le K_i
\]

The paper says queue lengths are normalized using logarithmic scaling. It gives the encoded state dimension as:

\[
\boxed{
\dim(s)=
\sum_{i=1}^{N}
\left\lceil\log_2(K_i+1)\right\rceil
}
\]

This is a logarithmic/bit-vector representation, not division by buffer capacity.

### 2.2 Project implementation

**Code location:** `src/wfq/env.py`, method `WFQEnv._state()`.

The repository uses:

```python
[
    length / self.capacity
    for length in self.system.lengths()
]
```

Thus each state component is:

\[
\boxed{
s_i=\frac{q_i}{K_i}
}
\]

For the project's queue capacity \(K_i=32\), each state component is the queue length divided by 32.

### 2.3 Comparison

| Item | Reference paper | Repository |
|---|---|---|
| State information | Queue lengths | Queue lengths |
| Normalization | Logarithmic/bit-vector encoding | Linear division by capacity |
| State dimension | Sum of \(\lceil\log_2(K_i+1)\rceil\) across queues | One floating-point value per queue |
| Eight queues, capacity 32 | \(8\lceil\log_2(33)\rceil=48\) encoded bits | 8 floating-point inputs |

**Conclusion:** The state information is conceptually the same, but the normalization/encoding is not. This is an implementation deviation from the paper. The completed experiment results correspond to the linear-normalized state actually used by the repository.

---

## 3. Action and Bandwidth Allocation

**Paper location:** Section 4.4, printed page 7; Section 5.1, printed page 18.

The paper defines the action as one positive bandwidth weight per queue, with all weights summing to one:

\[
\boxed{
a=(w_1,w_2,\ldots,w_N),\qquad
w_i>0,\qquad
\sum_{i=1}^{N}w_i=1
}
\]

The service rate for queue \(i\) is described in Section 4.1 as:

\[
\boxed{\mu_i=w_i b}
\]

Where \(b\) is the total system bandwidth.

For the eight-queue experiment, the paper specifies:

- \(b=1\) byte/second.
- Average packet size: 1 byte/packet.
- Therefore the total service capacity is 1 packet/second under this packet-size assumption.
- The eight weights must sum to 1.

### Project implementation

**Code location:** `src/wfq/env.py`, method `WFQEnv.step()`.

The environment checks that the action has one finite, positive weight per queue and normalizes it:

```python
normalized_weights = weights / np.sum(weights)
```

Mathematically:

\[
\boxed{
w_i^{\text{used}}=
\frac{\tilde w_i}{\sum_{j=1}^{N}\tilde w_j}
}
\]

Where \(\tilde w_i\) is the raw actor output.

**Conclusion:** The action constraint is enforced by normalization in the environment. The environment rejects zero or negative raw weights.

---

## 4. Fairness Target and Evaluation Metrics

**Paper location:** Section 4.4, printed page 9, Equations (4), (5), and (6).

### 4.1 Theoretical fairness target

The paper says the theoretical fairness allocation is calculated at each step from the queue state, specifically the queue lengths. It presents Equation (4) in terms of weights:

\[
\text{Fairness}_i=
\frac{w_i}{\sum_{j=1}^{N}w_j},
\qquad w_i\ge 0
\tag{4}
\]

The accompanying prose says the target allocation is based on the current queue state (queue length).

However, the printed equation as rendered is a normalization of weights and does not explicitly give a complete mapping from queue lengths \(q_i\) to target weights \(w_i\).

**Important limitation:** Do not replace this with an equal-share target \(1/N\), or claim that the paper explicitly defines \(w_i=q_i/\sum_jq_j\), unless another explicit source from the paper establishes that mapping. Equation (4) alone does not establish either claim. The exact queue-length-to-target-weight rule is under-specified in the paper's displayed equation and accompanying text.

### 4.2 RMSE

The paper defines the root mean square error between the policy's predicted weights and the theoretical fairness weights:

\[
\boxed{
\mathrm{RMSE}
=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(\hat y_i-y_i)^2
}
}
\tag{5}
\]

Where:

- \(\hat y_i\): Weight predicted by the policy network for queue \(i\).
- \(y_i\): Target weight computed by the paper's fairness function.
- \(N\): Number of queues.

Lower RMSE means the predicted allocation is closer to the target allocation.

### 4.3 MAE and MSE

The paper discusses MAE and MSE in its results, but the displayed fairness-metric equations transcribed here provide Equation (5) for RMSE and do not provide a separate numbered MAE or MSE equation in this section.

For completeness, the standard definitions over \(N\) queue-weight components are:

\[
\mathrm{MSE}
=
\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2
\]

\[
\mathrm{MAE}
=
\frac{1}{N}\sum_{i=1}^{N}|\hat y_i-y_i|
\]

And:

\[
\mathrm{RMSE}=\sqrt{\mathrm{MSE}}
\]

These are standard mathematical definitions. They should not be presented as separately numbered equations from the paper unless the original source explicitly numbers them that way.

### 4.4 Jain's Fairness Index

The paper gives:

\[
\boxed{
J(x)=
\frac{\left(\sum_{i=1}^{N}x_i\right)^2}
{N\sum_{i=1}^{N}x_i^2},
\qquad x_i\ge0
}
\tag{6}
\]

For non-negative values, the index is between \(1/N\) and 1 when at least one value is positive. A value of 1 means all values are equal.

**Interpretation caution:** The paper's prose describes Jain's index as indicating closeness to its theoretical fairness pattern, but the standard Jain formula applied to one vector measures equality among the entries of that vector. It does not by itself measure the distance between a predicted vector and a separate target vector. The paper's explanation of the numerator and denominator is imprecise. State exactly which vector is supplied to the implementation when reporting this metric.

### 4.5 Project implementation of Jain's index

**Code location:** `src/wfq/metrics.py`, function `jains_index()`.

The function implements:

```python
(total * total) / (n * sum_of_squares)
```

It rejects negative values, rejects an empty input, and returns 1.0 when all values are zero.

Equivalent formula:

\[
J(x)=\frac{(\sum_i x_i)^2}{N\sum_i x_i^2}
\]

This matches the standard Jain formula. It does not compute RMSE, MAE, or MSE in the displayed `metrics.py` implementation.

---

## 5. Soft Actor-Critic (SAC)

**Paper location:** Sections 4.3 to 4.5 and the SAC discussion in Section 4.5.

The paper describes two critics, target critics, entropy maximization, replay-buffer updates, and soft target updates.

SAC is an off-policy actor-critic algorithm. The actor represents a stochastic policy, while two critics estimate state-action values. The smaller target-critic estimate is used to reduce overestimation.

### 5.1 Soft Bellman target

The standard SAC target used to train the critics is:

\[
\boxed{
y_t=
r_t+
\gamma(1-d_t)
\left[
\min_{j\in\{1,2\}}
Q_{\bar\phi_j}(s_{t+1},a_{t+1})
-\alpha\log\pi_\theta(a_{t+1}\mid s_{t+1})
\right]
}
\]

Where:

- \(y_t\): Critic target.
- \(r_t\): Immediate reward.
- \(\gamma\): Discount factor.
- \(d_t\): Terminal indicator (1 if terminal, otherwise 0).
- \(s_{t+1}\): Next state.
- \(a_{t+1}\): Next action sampled from the actor.
- \(Q_{\bar\phi_1},Q_{\bar\phi_2}\): Target critics.
- \(\pi_\theta\): Stochastic actor policy.
- \(\alpha\): Entropy temperature.

The entropy term rewards policy diversity. The minimum of the two target critics reduces optimistic value estimates.

This is the standard SAC target equation written with an explicit terminal mask. It is a mathematical explanation of the algorithm, not a claim that this exact notation is printed as an equation in the paper.

### 5.2 Critic loss

For each critic \(Q_{\phi_j}\), the standard squared-error objective is:

\[
\boxed{
L_Q(\phi_j)=
\mathbb E_{(s_t,a_t,r_t,s_{t+1},d_t)\sim\mathcal D}
\left[
\left(Q_{\phi_j}(s_t,a_t)-y_t\right)^2
\right]
}
\]

Where \(\mathcal D\) is the replay buffer.

### 5.3 Actor objective

A common SAC actor objective is:

\[
\boxed{
L_\pi(\theta)=
\mathbb E_{s_t\sim\mathcal D,\;a_t\sim\pi_\theta}
\left[
\alpha\log\pi_\theta(a_t\mid s_t)
-\min_{j\in\{1,2\}}Q_{\phi_j}(s_t,a_t)
\right]
}
\]

Minimizing this objective encourages actions with high estimated value while retaining entropy.

### 5.4 Target-network soft update

The target critics are updated slowly:

\[
\boxed{
\bar\phi_j\leftarrow
\tau\phi_j+(1-\tau)\bar\phi_j
}
\]

Where \(\phi_j\) are online critic parameters and \(\bar\phi_j\) are target critic parameters.

### 5.5 Project hyperparameters

The project's `config/base.yaml` records:

| Parameter | Project value |
|---|---:|
| Discount factor \(\gamma\) | 0.99 |
| Entropy coefficient \(\alpha\) | 0.2 |
| Target update rate \(\tau\) | 0.01 |
| Actor learning rate | 0.0001 |
| Critic learning rate | 0.001 |
| Hidden units | 64 |
| Replay buffer size | 10,000 |
| Batch size | 128 |

These values match the corresponding eight-queue settings in the paper's Table 1 where the same parameter is listed.

---

## 6. Elastic Weight Consolidation (EWC)

**Paper location:** Section 4.3, printed page 8, Equations (2) and (3); refresh schedule in Section 5.1, printed page 18.

EWC adds a quadratic penalty to discourage important network parameters from moving too far from previously learned values.

### 6.1 EWC-regularized loss

The paper's Equation (2) is:

\[
\boxed{
L_{\mathrm{total}}
=
L_{\mathrm{RL}}
+
\frac{\lambda_{\mathrm{EWC}}}{2}
\sum_i F_i(\theta_i-\theta_i^*)^2
}
\tag{2}
\]

Where:

- \(L_{\mathrm{RL}}\): Ordinary reinforcement-learning loss.
- \(\lambda_{\mathrm{EWC}}\): Regularization strength.
- \(F_i\): Fisher importance estimate for parameter \(i\).
- \(\theta_i\): Current parameter.
- \(\theta_i^*\): Saved parameter value (anchor).

A larger Fisher value gives that parameter a stronger penalty for moving away from its anchor.

The paper says the EWC term is added to both policy and critic losses.

### 6.2 Fisher Information estimate

The paper's Equation (3) gives the Fisher estimate in the form:

\[
\boxed{
F_\theta=
\mathbb E
\left[
\left(
\frac{\partial\log\pi_\theta(a\mid s)}
{\partial\theta}
\right)^2
\right]
}
\tag{3}
\]

The paper describes estimating Fisher values by computing squared gradients and averaging them over samples from the replay buffer. It also says the procedure is applied to the critic network \(Q\).

The expression above is the policy-gradient form shown in Equation (3). The paper's prose extends the procedure to the critic, but does not provide a separate fully expanded critic-specific Fisher equation in this section.

### 6.3 Anchor and Fisher refresh

The saved anchor is the parameter snapshot used as \(\theta^*\). The Fisher values estimate which parameters are important for the previously learned task.

For the eight-queue experiment, the paper states that the Fisher matrix and saved parameters are refreshed every 30 episodes.

### 6.4 Project settings and implementation notes

The project's `config/base.yaml` specifies:

| Parameter | Project value |
|---|---:|
| EWC coefficient \(\lambda\) | 2271 |
| Fisher refresh interval | 30 episodes |
| Fisher batches | 5 |
| EWC mode | `paper` |

The value 2271 is the paper's eight-queue value. The five Fisher mini-batches are an implementation setting recorded in this project's configuration; do not attribute the value five to the paper unless a paper passage explicitly supports it.

The repository has two EWC modes:

- `paper`: Refreshes the anchor and Fisher estimate on the configured schedule.
- `task_anchored`: Uses the project's alternative frozen-anchor design for the traffic-switch experiment.

The task-anchored mode is a project extension, not the paper's every-30-episode refresh procedure.

---

## 7. Project Configuration and Paper Comparison

The final eight-queue configuration in `config/base.yaml` is:

| Setting | Project value | Paper comparison |
|---|---:|---|
| Number of queues | 8 | Eight-queue experiment |
| Arrival rates (packets/s) | 0.03, 0.16, 0.20, 0.10, 0.08, 0.15, 0.06, 0.05 | Same values as Section 5.1 |
| Total bandwidth | 1.0 packet/s in this simulator's units | Paper: 1 byte/s with 1 byte packets |
| SCV | 0.8 | Same as Section 5.1 |
| Queue capacity | 32 packets per queue | Same as Section 5.1 |
| Decision interval | 20 seconds | Same as Section 5.1 |
| Episode duration | 2,000 seconds | Same as Section 5.1 |
| Decisions per episode | 100 | Follows from 2,000 / 20 |
| Training episodes | 1,000 | Same as Section 5.1 |
| Hidden units | 64 | Eight-queue setting |
| Replay buffer | 10,000 | Same as Table 1 |
| Batch size | 128 | Same as Table 1 |
| EWC coefficient | 2271 | Same as Table 1 |
| Fisher refresh interval | 30 episodes | Same as Section 5.1 |

The project configuration and the paper are not identical in every respect. In particular, the state normalization and holding-cost sampling method differ as described above. The project also includes additional baselines and a traffic-switch experiment.

---

## 8. Verification Summary

| No. | Item | Verified conclusion |
|---:|---|---|
| 1 | Reward equation (1) | Paper equation transcribed |
| 2 | Q1: Holding-cost coefficient | \(h_i=1/N\), confirmed in Section 5, printed p.17 |
| 3 | Q2: Packet-loss coefficient | \(c_i=1\), confirmed in Section 5, printed p.17 |
| 4 | Q3: Holding-cost time basis | Paper says per unit time; 20 s is action interval; exact interval integration is not specified |
| 5 | Reward implementation | Coefficients match; code uses end-of-interval queue snapshot and interval loss count |
| 6 | State normalization | Paper logarithmic encoding; code linear division by capacity |
| 7 | Action constraint | Positive weights normalized to sum to one |
| 8 | Fairness target | Described as queue-state-dependent; displayed Eq. (4) does not fully specify queue-length-to-target mapping |
| 9 | RMSE | Equation (5) transcribed |
| 10 | MAE and MSE | Standard definitions documented; no separate numbered formulas established here |
| 11 | Jain's index | Equation (6) and repository implementation match standard formula |
| 12 | SAC | Standard SAC target, critic and actor objectives documented; parameters recorded from Table 1 |
| 13 | EWC penalty | Equation (2) transcribed |
| 14 | Fisher estimate | Equation (3) transcribed; paper describes squared-gradient averaging |
| 15 | EWC refresh | Every 30 episodes in paper's eight-queue experiment |

---

## 9. Viva-Ready Answers: Q1, Q2 and Q3

**Q1. Is holding cost divided by the number of queues?**

Yes. The paper specifies \(h_i=1/N\). With eight queues, each queue's holding-cost coefficient is \(1/8\). Therefore, the total holding term is the sum of the queue lengths divided by eight.

**Q2. Is packet loss also divided by the number of queues?**

No. The paper specifies \(c_i=1\). The loss penalty is the direct sum of lost packets across all queues. There is no division by the number of queues in that term.

**Q3. Is holding cost calculated every second or every 20 seconds?**

The paper defines holding cost per unit of time and separately says the agent chooses a new action every 20 seconds. It does not clearly state that Equation (1) integrates queue length across the entire 20-second interval. Our code advances the simulator for 20 seconds and then samples queue lengths once at the end of the interval. Therefore, our implementation is a snapshot-based discrete approximation, not a continuous-time integral.

---

## References

1. Mawlood, M. A., and Mahmood, D. A. (2025). "Optimizing Weighted Fair Queuing with Deep Reinforcement Learning for Dynamic Bandwidth Allocation." *Telecom*, 6(3), 46. https://doi.org/10.3390/telecom6030046. Relevant locations: Section 4.1 (queue model and service rate), Section 4.4 (state, action and reward, printed pp. 7-8), Section 4.5 (fairness and error metrics, printed pp. 9-10), and Section 5 / Table 1 (experiment settings and reward coefficients, printed pp. 17-18).

2. Repository implementation: `src/wfq/env.py`, `src/wfq/metrics.py`, `src/wfq/baselines.py`, and `config/base.yaml`.
