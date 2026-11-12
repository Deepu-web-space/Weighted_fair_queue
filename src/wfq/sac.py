"""SAC agent, with an optional EWC memory penalty (Day 3, Blocks 2 and 3).

One update does four things:
1. Critics learn to predict the target value y.
2. The actor learns to choose weights that score well under the critics,
   with an entropy bonus that keeps its choices varied.
3. If EWC is on, both losses get a penalty for moving important weights.
4. Target critics move slowly toward the critics (soft update).

EWC is off when no ewc_config is given, or when its lambda is 0. In that
case no extra random numbers are drawn, so the run matches plain SAC exactly.

Every number comes from config/base.yaml. Nothing is typed in here, except
the plain structure of the formulas (for example the 1.0 in 1.0 - done).
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from wfq.ewc import EWCAnchor
from wfq.networks import Actor, Critic
from wfq.replay import ReplayBuffer


class SACAgent:
    def __init__(self, sim_config, train_config, seed, ewc_config=None):
        n = sim_config["n_queues"]
        hidden = train_config["hidden_units"]

        self.n_queues = n
        self.gamma = train_config["gamma"]
        self.tau = train_config["tau"]
        self.alpha = train_config["alpha"]
        self.batch_size = train_config["batch_size"]
        self.warmup_steps = train_config["warmup_steps"]
        self.min_weight = train_config["min_weight"]

        # The seed sets torch's random numbers (network start, action
        # sampling) and numpy's (replay batches). Environment seeds are separate.
        torch.set_num_threads(train_config["torch_threads"])
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.actor = Actor(n, hidden, train_config["min_concentration"])
        self.critic1 = Critic(n, hidden)
        self.critic2 = Critic(n, hidden)

        # Target critics start as exact copies and are never trained directly.
        self.target1 = copy.deepcopy(self.critic1)
        self.target2 = copy.deepcopy(self.critic2)
        for p in list(self.target1.parameters()) + list(self.target2.parameters()):
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=train_config["lr_actor"]
        )
        self.critic_opt = torch.optim.Adam(
            self._critic_params(), lr=train_config["lr_critic"]
        )

        self.buffer = ReplayBuffer(train_config["replay_size"], n, n)

        # EWC is on only when a config is given and lambda is above zero.
        self.ewc_on = ewc_config is not None and ewc_config["lambda"] > 0
        if self.ewc_on:
            self.fisher_batches = ewc_config["fisher_batches"]
            self.ewc_actor = EWCAnchor(ewc_config["lambda"])
            self.ewc_critic = EWCAnchor(ewc_config["lambda"])

    def _critic_params(self):
        return list(self.critic1.parameters()) + list(self.critic2.parameters())

    def _ready(self):
        return self.buffer.size >= max(self.warmup_steps, self.batch_size)

    def _batch(self):
        s, a, r, s2, d = self.buffer.sample(self.batch_size, self.rng)
        return (
            torch.as_tensor(s),
            torch.as_tensor(a),
            torch.as_tensor(r),
            torch.as_tensor(s2),
            torch.as_tensor(d),
        )

    def _floor(self, w):
        """Keep every weight above min_weight, then rescale to sum to 1.

        The environment refuses zero weights, and log-probabilities
        become undefined at zero.
        """
        w = torch.clamp(w, min=self.min_weight)
        return w / w.sum(dim=-1, keepdim=True)

    def act(self, state, deterministic=False):
        """Return a weight vector (numpy array, sums to 1) for one state."""
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                w = self.actor.mean_weights(s)
            else:
                w = self._floor(self.actor(s).sample())
        return w.squeeze(0).numpy()

    def store(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def _critic_loss(self, batch):
        """Base critic loss (without EWC): both critics move toward y."""
        s, a, r, s2, d = batch

        # Critic target: reward plus the discounted future value,
        # using the smaller of the two target critics and the entropy bonus.
        with torch.no_grad():
            dist2 = self.actor(s2)
            a2 = self._floor(dist2.rsample())
            logp2 = dist2.log_prob(a2).unsqueeze(-1)
            q_next = torch.min(self.target1(s2, a2), self.target2(s2, a2))
            y = r + self.gamma * (1.0 - d) * (q_next - self.alpha * logp2)

        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        return F.mse_loss(q1, y) + F.mse_loss(q2, y)

    def _actor_loss(self, s):
        """Base actor loss (without EWC): prefer weights the critics score highly."""
        dist = self.actor(s)
        a_new = self._floor(dist.rsample())
        logp = dist.log_prob(a_new).unsqueeze(-1)
        q_new = torch.min(self.critic1(s, a_new), self.critic2(s, a_new))
        return (self.alpha * logp - q_new).mean()

    def update(self):
        """One gradient step. Returns None during warm-up, else the losses."""
        if not self._ready():
            return None

        batch = self._batch()
        s = batch[0]

        # 1. Critics: base loss plus the EWC penalty when EWC is on.
        critic_loss = self._critic_loss(batch)
        if self.ewc_on:
            critic_penalty = self.ewc_critic.penalty(self._critic_params())
            total = critic_loss + critic_penalty
        else:
            critic_penalty = torch.zeros(())
            total = critic_loss
        self.critic_opt.zero_grad()
        total.backward()
        self.critic_opt.step()

        # 2. Actor: base loss plus the EWC penalty when EWC is on.
        #    Gradients reach the critics too, but they are cleared
        #    before the next critic step.
        actor_loss = self._actor_loss(s)
        if self.ewc_on:
            actor_penalty = self.ewc_actor.penalty(list(self.actor.parameters()))
            total = actor_loss + actor_penalty
        else:
            actor_penalty = torch.zeros(())
            total = actor_loss
        self.actor_opt.zero_grad()
        total.backward()
        self.actor_opt.step()

        # 3. Move the target critics slowly toward the critics.
        self._soft_update(self.critic1, self.target1)
        self._soft_update(self.critic2, self.target2)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_penalty": float(critic_penalty.item()),
            "actor_penalty": float(actor_penalty.item()),
        }

    def refresh_ewc(self):
        """Save new Fisher values and a new anchor for both networks.

        Does nothing when EWC is off, so lambda 0 draws no extra random numbers.
        """
        if not self.ewc_on:
            return
        if not self._ready():
            raise RuntimeError(
                "Too few transitions for a Fisher refresh; wait until warm-up ends"
            )
        self.ewc_critic.snapshot(
            self._critic_params(),
            lambda: self._critic_loss(self._batch()),
            self.fisher_batches,
        )
        self.ewc_actor.snapshot(
            self.actor.parameters(),
            lambda: self._actor_loss(self._batch()[0]),
            self.fisher_batches,
        )

    def _soft_update(self, source, target):
        """target = tau * source + (1 - tau) * target."""
        with torch.no_grad():
            for sp, tp in zip(source.parameters(), target.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * sp)
