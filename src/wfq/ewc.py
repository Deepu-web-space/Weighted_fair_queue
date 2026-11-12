"""EWC memory: a penalty that keeps important weights near where they were (Day 3, Block 3).

At each refresh two things are saved for one network:
- Fisher values: for each weight, how much the learning loss depended on it.
  Found by averaging squared gradients over several mini-batches.
  A large value means "this weight mattered".
- Anchor: a copy of every weight at that moment.

The penalty is (lambda / 2) * sum of Fisher * (weight - anchor) squared.
It is zero right after a refresh. It grows when an important weight moves
away from its anchor. Before the first refresh nothing is saved, so it is zero.
"""

import torch


class EWCAnchor:
    def __init__(self, lambda_ewc):
        self.lambda_ewc = float(lambda_ewc)
        self.fisher = None
        self.anchor = None

    def has_snapshot(self):
        return self.fisher is not None

    def snapshot(self, params, loss_fn, n_batches):
        """Estimate Fisher values, then save the anchor.

        loss_fn() must return a scalar loss on a fresh mini-batch.
        torch.autograd.grad returns gradients without touching the
        optimiser's own gradient fields.
        """
        if n_batches < 1:
            raise ValueError("n_batches must be at least 1")

        params = list(params)
        totals = [torch.zeros_like(p) for p in params]
        for _ in range(n_batches):
            loss = loss_fn()
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            for total, g in zip(totals, grads):
                if g is not None:
                    total += g.detach() ** 2

        self.fisher = [t / n_batches for t in totals]
        self.anchor = [p.detach().clone() for p in params]

    def penalty(self, params):
        """(lambda / 2) * sum Fisher * (weight - anchor)^2. Zero before any snapshot.

        params must be in the same order as in snapshot().
        """
        if not self.has_snapshot():
            return torch.zeros(())

        total = torch.zeros(())
        for p, f, a in zip(params, self.fisher, self.anchor):
            total = total + (f * (p - a) ** 2).sum()
        return 0.5 * self.lambda_ewc * total
