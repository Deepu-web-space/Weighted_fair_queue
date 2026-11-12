# Paper Equations and Mathematical Definitions

This document records the mathematical equations used in the Weighted Fair Queueing (WFQ) project.

It covers the reward function, fair-share heuristic, Soft Actor-Critic (SAC), and Elastic Weight Consolidation (EWC).

The equations are separated from implementation details so that the code can be checked against the research paper.

---

## 1. Equation (1): Reward Function

**Location in paper:** Section 3, under the heading "Reward:", and repeated in Section 5.

### 1.1 Mathematical Equation

\[
\boxed{
r_t=-\sum_{i=1}^{N}(h_iq_i+c_il_i)
}
\]

Where:

- \(r_t\): Reward at decision step \(t\).
- \(N\): Total number of queues (8 for Experiment 1).
- \(h_i\): Holding-cost coefficient for queue \(i\).
- \(q_i\): Queue length for queue \(i\).
- \(c_i\): Packet-loss penalty coefficient for queue \(i\).
- \(l_i\): Number of packets lost from queue \(i\).

The negative sign indicates that queue lengths and packet losses contribute negatively to the reward.

### 1.2 Questions Requiring Verification from the Paper

**Q1. Is the holding cost divided by N?**

Answer: To be verified.

Paper's mathematical definition:

\[
h_i=?
\]

Check whether the paper specifies \(h_i=1/N\) or another value.

**Q2. Is the loss penalty divided by N or summed directly?**

Answer: To be verified.

Check the definition of \(c_i\) in Sections 3 and 5.

**Q3. Is holding cost calculated per second or per decision step?**

Answer: To be verified.

Check whether the paper defines queue holding cost over continuous time or at discrete decision intervals.

### 1.3 Current Code Implementation

Location: `src/wfq/env.py`

```python
holding_cost = sum(queue_lengths) / self.n_queues

loss_penalty = sum(lost_counts)

reward = -(holding_cost + loss_penalty)
```

The implemented mathematical expression is:

\[
r_t=
-\left(
\frac{1}{N}\sum_{i=1}^{N}q_i
+
\sum_{i=1}^{N}l_i
\right)
\]

### 1.4 Comparison with the Paper

The implementation matches Equation (1) if:

\[
h_i=\frac{1}{N}
\]

and

\[
c_i=1
\]

for every queue.

However, the time interpretation of \(q_i\), and whether it represents instantaneous queue length or time-integrated holding cost, must also be verified.

**Current verification status: Pending original paper inspection.**

---

## 2. Fair-Share Heuristic Target

**Location in paper:** Section 5, heuristic baseline used for RMSE, MAE, MSE and Jain's fairness index.

### 2.1 Mathematical Definition

The fair-share target is represented as:

\[
w_{\text{fair},i} = ?
\]

The exact target allocation must be extracted from the heuristic definition in the paper.

If the paper specifies equal bandwidth allocation among all queues, the expression is:

\[
\boxed{
w_{\text{fair},i}=\frac{1}{N}
}
\]

For eight queues:

\[
w_{\text{fair},i}=\frac{1}{8}=0.125
\]

This equal-allocation expression must not be assumed to be the paper's heuristic until verified.

### 2.2 Purpose

The fair-share target provides a reference allocation against which the learned policy can be evaluated.

The paper uses the heuristic baseline for comparison through:

- Root Mean Square Error (RMSE).
- Mean Absolute Error (MAE).
- Mean Square Error (MSE).
- Jain's Fairness Index.

### 2.3 Verification Status

Pending confirmation of the exact heuristic allocation used in Tables 3 and 4.

---

## 3. SAC Critic Target Equation

**Location in paper:** Section 4.1.

### 3.1 Mathematical Equation

\[
\boxed{
y_t =
r_t+
\gamma
\left[
\min_{j\in\{1,2\}}
Q_{\bar{\phi}_j}(s_{t+1},a_{t+1})
-
\alpha\log\pi(a_{t+1}|s_{t+1})
\right]
}
\]

Where:

- \(y_t\): Target value for critic training.
- \(r_t\): Immediate reward.
- \(\gamma\): Discount factor.
- \(s_{t+1}\): Next state.
- \(a_{t+1}\): Next action sampled from the policy.
- \(Q_{\bar{\phi}_1}\), \(Q_{\bar{\phi}_2}\): Target critic networks.
- \(\pi\): Stochastic policy.
- \(\alpha\): Entropy temperature coefficient.

The minimum of the two target critics reduces overestimation bias.

The entropy term encourages exploration.

### 3.2 Project Parameters

According to Table 1:

| Parameter | Value |
|---|---:|
| Discount factor (\(\gamma\)) | 0.99 |
| Entropy coefficient (\(\alpha\)) | 0.2 |
| Soft update coefficient (\(\tau\)) | 0.01 |

### 3.3 Target Network Update

The target critic parameters are updated using:

\[
\boxed{
\bar{\phi}
\leftarrow
\tau\phi+(1-\tau)\bar{\phi}
}
\]

Where:

- \(\phi\): Online critic parameters.
- \(\bar{\phi}\): Target critic parameters.
- \(\tau\): Soft update coefficient.

For this project:

\[
\tau=0.01
\]

---

## 4. EWC Penalty Equation

**Location in paper:** Sections 4.2 and 4.3.

### 4.1 Mathematical Equation

\[
\boxed{
L_{\text{EWC}}
=
L_{\text{SAC}}
+
\frac{\lambda}{2}
\sum_i
F_i(\theta_i-\theta_i^*)^2
}
\]

Where:

- \(L_{\text{EWC}}\): Total loss with EWC regularization.
- \(L_{\text{SAC}}\): Original SAC loss.
- \(\lambda\): EWC regularization coefficient.
- \(F_i\): Diagonal element of the Fisher Information Matrix.
- \(\theta_i\): Current network parameter.
- \(\theta_i^*\): Previously saved parameter value (anchor).

### 4.2 Purpose

EWC helps prevent catastrophic forgetting during continual learning.

It penalizes changes to parameters that were important for previously learned tasks.

Parameters with larger Fisher Information values receive stronger protection.

### 4.3 Project Parameters

According to Table 1:

| Parameter | Value |
|---|---:|
| EWC coefficient (\(\lambda\)) | 2271 |
| Number of queues | 8 |
| Anchor refresh interval | 30 episodes |

### 4.4 Anchor Update

The reference parameters are stored as:

\[
\theta^*=\theta_{\text{saved}}
\]

The Fisher Information Matrix estimates the importance of individual parameters.

According to Section 4.3, the parameter anchors are refreshed every 30 episodes.

**Implementation note:** The exact Fisher estimation procedure and which network parameters receive EWC regularization must follow Sections 4.2 and 4.3.

---

## 5. Mathematical Verification Checklist

The following items must be confirmed against the original research paper before finalizing the implementation.

| No. | Item | Status |
|---|---|---|
| 1 | Exact reward Equation (1) | Equation transcribed; coefficients pending |
| 2 | Holding-cost coefficient | Pending |
| 3 | Packet-loss penalty coefficient | Pending |
| 4 | Holding-cost time interpretation | Pending |
| 5 | State normalization method | Pending |
| 6 | Fair-share heuristic target | Pending |
| 7 | RMSE definition | Pending |
| 8 | MAE definition | Pending |
| 9 | MSE definition | Pending |
| 10 | Jain's Fairness Index | Pending |
| 11 | SAC critic target | Transcribed |
| 12 | EWC penalty | Transcribed |
| 13 | Fisher Information estimation | Pending |

---

## 6. Implementation Consistency

The equations documented here will serve as the mathematical reference for:

1. WFQ environment implementation.
2. SAC agent implementation.
3. EWC regularization.
4. Continual learning experiments.
5. Evaluation metrics and comparison with heuristic baselines.

Any difference between the paper's mathematical formulation and the implementation must be documented and justified before experimental training.